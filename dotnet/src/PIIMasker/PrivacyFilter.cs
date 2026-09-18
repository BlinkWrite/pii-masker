using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using Tokenizers.DotNet;

namespace PIIMasker;

/// <summary>Masked fields plus the map that puts the originals back.</summary>
/// <param name="Fields">The input fields with every detected span replaced by a placeholder.</param>
/// <param name="Restore">Placeholder → original text, for <see cref="PrivacyFilter.Restore"/>.</param>
public sealed record PrivacyMaskResult(
    IReadOnlyList<string> Fields, IReadOnlyDictionary<string, string> Restore);

/// <summary>The masking seam, so a host can substitute a filter in its own tests.</summary>
/// <remarks>
/// Deliberately one method. A host should not be able to fake "masking succeeded" through a wider
/// surface than the real filter offers — the whole point of the gate is that there is one way past it.
/// </remarks>
public interface IPrivacyFilter
{
    /// <summary>Mask a set of fields in one pass, or return null having masked nothing.</summary>
    /// <param name="fields">The fields to mask.</param>
    /// <param name="cancellationToken">Cancels the pass.</param>
    Task<PrivacyMaskResult?> MaskAsync(IReadOnlyList<string> fields, CancellationToken cancellationToken = default);
}

/// <summary>One span the detector claimed, in character offsets.</summary>
/// <param name="Start">Character offset of the span's first character.</param>
/// <param name="Length">Span length in characters.</param>
/// <param name="Label">The label the model filed it under.</param>
/// <param name="Confidence">Sigmoid probability, compared against <see cref="MaskerConfig.Threshold"/>.</param>
public readonly record struct DetectedEntity(int Start, int Length, string Label, float Confidence);

/// <summary>Detects PII on-device and replaces it with reversible placeholders.</summary>
/// <remarks>
/// <para>
/// Fails closed on every path. A load failure, a hash mismatch, a timeout, a cancellation, input
/// over the token budget, or a round trip that does not come back intact all return null rather
/// than partly-masked text — because the caller's next move is to send it somewhere.
/// </para>
/// <para>
/// One masking pass is serialized behind <see cref="MaskAsync"/>'s gate: the session is not
/// re-entrant, and a queue is cheaper than a second model in memory.
/// </para>
/// </remarks>
public sealed class PrivacyFilter : IPrivacyFilter, IDisposable
{
    /// <summary>ASCII RECORD SEPARATOR — the character the guard and separator are built from.</summary>
    /// <remarks>
    /// Spelled as a numeric char constant rather than a string escape on purpose. The literal
    /// character is invisible in an editor and does not survive a copy-paste, and an escape sequence
    /// is one careless tool away from being expanded into that invisible character — both of which
    /// happened while this file was being written. Chosen as the delimiter because it is a control
    /// character no user types.
    /// </remarks>
    private const char RecordSeparator = (char)0x1E;

    // `static readonly` rather than `const`: C# will not fold string + char into a constant
    // expression, and spelling these as escapes is what this file is deliberately avoiding.
    // Nothing here needs a compile-time constant — they are only ever joined, split and compared.

    /// <summary>Wraps the joined blob so a truncated round trip is detectable.</summary>
    public static readonly string MaskGuard = new(RecordSeparator, 2);

    /// <summary>Separates fields inside the joined blob.</summary>
    public static readonly string MaskSeparator = $" {new string(RecordSeparator, 3)} ";

    private readonly string modelDirectory;
    private readonly ModelPin pin;
    private readonly MaskerConfig config;
    private readonly MaskerLogging logging;
    private readonly TimeSpan timeout;
    private readonly SemaphoreSlim inferenceGate = new(1, 1);
    private readonly object warmUpLock = new();
    private Tokenizer? tokenizer;
    private InferenceSession? session;
    private Task<bool>? warmUpTask;

    /// <summary>Why the warm-up failed, when it did. Null before a warm-up and after a good one.</summary>
    public Exception? WarmUpFailure { get; private set; }

    /// <param name="modelDirectory">The directory holding the three model files.</param>
    /// <param name="pin">The model identity to enforce. Defaults to <see cref="ModelPin.Current"/>.</param>
    /// <param name="config">Detection knobs. Defaults to <see cref="MaskerConfig.Default"/>.</param>
    /// <param name="logging">Where diagnostics go. Defaults to silent.</param>
    /// <param name="timeout">Budget for one masking call. Defaults to four seconds.</param>
    public PrivacyFilter(
        string modelDirectory,
        ModelPin? pin = null,
        MaskerConfig? config = null,
        MaskerLogging? logging = null,
        TimeSpan? timeout = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(modelDirectory);
        this.modelDirectory = modelDirectory;
        this.pin = pin ?? ModelPin.Current;
        this.config = config ?? MaskerConfig.Default;
        this.logging = logging ?? MaskerLogging.Silent;
        this.timeout = timeout ?? TimeSpan.FromSeconds(4);
    }

    /// <summary>Build the tokenizer and session off the caller's path.</summary>
    /// <remarks>
    /// So the first real request does not spend its masking deadline loading the model. Concurrent
    /// callers share one load attempt.
    /// </remarks>
    /// <param name="cancellationToken">Abandons waiting; the load itself continues.</param>
    public Task<bool> WarmUpAsync(CancellationToken cancellationToken = default)
    {
        Task<bool> task;
        lock (warmUpLock) task = warmUpTask ??= Task.Run(WarmUpCoreAsync, CancellationToken.None);
        return task.WaitAsync(cancellationToken);
    }

    private async Task<bool> WarmUpCoreAsync()
    {
        await inferenceGate.WaitAsync().ConfigureAwait(false);
        try
        {
            EnsureLoaded();
            ModelStore.MarkLoaded();
            return true;
        }
        catch (Exception exception)
        {
            WarmUpFailure = exception;
            // Recorded on the store, not just here: nothing else can tell "the files are present"
            // from "the files are present and are not a model", so without this the host looks
            // healthy while masking nothing, with no affordance to recover.
            ModelStore.MarkUnusable();
            logging.Note($"privacy: model {pin.Version} warm-up failed: {exception.GetType().Name}");
            return false;
        }
        finally { inferenceGate.Release(); }
    }

    /// <summary>Mask a set of fields in one pass, or return null having sent nothing.</summary>
    /// <param name="fields">The fields to mask. Masked together so one inference covers them all.</param>
    /// <param name="cancellationToken">Cancels the pass; a cancelled pass masks nothing.</param>
    public async Task<PrivacyMaskResult?> MaskAsync(
        IReadOnlyList<string> fields, CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(fields);
        var started = Stopwatch.GetTimestamp();
        if (!await inferenceGate.WaitAsync(timeout, cancellationToken).ConfigureAwait(false))
        {
            LogTimeout();
            return null;
        }
        try
        {
            var remaining = timeout - Stopwatch.GetElapsedTime(started);
            if (remaining <= TimeSpan.Zero) { LogTimeout(); return null; }

            using var runOptions = new RunOptions();
            var operation = Task.Run(() => Mask(fields, runOptions, cancellationToken), CancellationToken.None);
            var timeoutTask = Task.Delay(remaining, cancellationToken);
            if (await Task.WhenAny(operation, timeoutTask).ConfigureAwait(false) == operation)
            {
                var result = await operation.ConfigureAwait(false);
                cancellationToken.ThrowIfCancellationRequested();
                if (Stopwatch.GetElapsedTime(started) <= timeout) return result;
                LogTimeout();
                return null;
            }

            // Terminate rather than abandon: the run holds the gate, and leaving it in flight would
            // stall every queued request behind a pass whose answer is already discarded.
            runOptions.Terminate = true;
            try { await operation.ConfigureAwait(false); } catch (Exception) { /* already discarded */ }
            cancellationToken.ThrowIfCancellationRequested();
            LogTimeout();
            return null;
        }
        catch (OperationCanceledException) { throw; }
        catch (InvalidDataException) { return null; }
        catch (Exception exception)
        {
            logging.Note($"privacy: maskFields failed: {exception.GetType().Name}");
            return null;
        }
        finally { inferenceGate.Release(); }
    }

    private PrivacyMaskResult Mask(
        IReadOnlyList<string> fields, RunOptions runOptions, CancellationToken cancellationToken)
    {
        if (fields.Count == 0) return new([], new Dictionary<string, string>(StringComparer.Ordinal));

        var joined = JoinForMasking(fields);
        // Caller input containing the separator would cross field boundaries on the way back,
        // attributing one field's text to another. Refuse before inference rather than detect it after.
        var initial = SplitFields(joined, fields.Count);
        if (initial == null || !initial.SequenceEqual(fields, StringComparer.Ordinal))
        {
            logging.Note("privacy: maskFields input contains a field separator — dropping");
            throw new InvalidDataException("Privacy fields contain the structural separator.");
        }
        EnsureLoaded();

        var restore = new Dictionary<string, string>(StringComparer.Ordinal);
        var counts = new Dictionary<string, int>(StringComparer.Ordinal);
        var detections = DropProtectedSpans(joined, Detect(joined, runOptions, cancellationToken));
        var masked = Apply(joined, detections, restore, counts);
        var split = SplitFields(masked, fields.Count);
        if (split == null)
        {
            logging.Note(
                $"privacy: maskFields integrity check failed for {fields.Count} fields — separator or guard changed; dropping");
            throw new InvalidDataException("Privacy field integrity check failed.");
        }
        logging.TraceText(() => $"privacy: masked {fields.Count} fields, {restore.Count} placeholders: {masked}");
        return new(split, restore);
    }

    private void LogTimeout() =>
        logging.Note($"privacy: maskFields exceeded {timeout.TotalSeconds:F1}s timeout — dropping");

    /// <summary>Join several fields into the one blob a single inference pass sees.</summary>
    /// <remarks>
    /// Guard, separator, fields, separator, guard — so a truncated or rewritten round trip is
    /// detectable rather than silently producing the wrong number of fields. Internal rather than
    /// public: a caller batches by handing <see cref="MaskAsync"/> a list, and the framing is this
    /// type's business. The Swift target exposes its counterpart because its tests reach it that
    /// way; here the test assembly is a friend.
    /// </remarks>
    internal static string JoinForMasking(IEnumerable<string> fields) =>
        string.Join(MaskSeparator, new[] { MaskGuard }.Concat(fields).Append(MaskGuard));

    /// <summary>Recover the fields from a joined blob, or null when the framing did not survive.</summary>
    internal static string[]? SplitFields(string joined, int count)
    {
        var parts = joined.Split(MaskSeparator, StringSplitOptions.None);
        return parts.Length == count + 2 && parts[0] == MaskGuard && parts[^1] == MaskGuard
            ? parts[1..^1] : null;
    }

    /// <summary>Exclude the structural markers from detections, splitting spans that cross one.</summary>
    /// <remarks>
    /// The detector can false-positive a marker as PII, and masking it would corrupt the round trip.
    /// A crossing span is SPLIT rather than dropped: dropping the whole span would leave the
    /// detected secret on either side of the marker unmasked, which is the failure this guards.
    /// </remarks>
    internal static IEnumerable<DetectedEntity> DropProtectedSpans(
        string text, IEnumerable<DetectedEntity> detections)
    {
        var markers = MarkerPattern.Matches(text).Cast<Match>().ToArray();
        foreach (var detection in detections)
        {
            var pieces = new List<(int Start, int End)> { (detection.Start, detection.Start + detection.Length) };
            foreach (var marker in markers)
            {
                var next = new List<(int Start, int End)>();
                foreach (var piece in pieces)
                {
                    if (marker.Index >= piece.End || marker.Index + marker.Length <= piece.Start)
                        next.Add(piece);
                    else
                    {
                        if (piece.Start < marker.Index) next.Add((piece.Start, marker.Index));
                        if (piece.End > marker.Index + marker.Length)
                            next.Add((marker.Index + marker.Length, piece.End));
                    }
                }
                pieces = next;
            }
            foreach (var piece in pieces)
                yield return detection with { Start = piece.Start, Length = piece.End - piece.Start };
        }
    }

    // Two or three separators in a row: the guard contributes two, a field boundary three.
    private static readonly Regex MarkerPattern =
        new($"{RecordSeparator}{{2,3}}", RegexOptions.Compiled);
    private static readonly Regex NonAlphanumeric = new("[^A-Z0-9]+", RegexOptions.Compiled);
    private static readonly Regex Words = new(@"\S+", RegexOptions.Compiled);

    /// <summary>Replace detected spans with numbered placeholders, recording how to undo it.</summary>
    /// <param name="text">The text to mask.</param>
    /// <param name="detections">Spans to replace.</param>
    /// <param name="restore">Receives placeholder → original.</param>
    /// <param name="counts">Receives per-label counters, so placeholders number from 1.</param>
    public static string Apply(
        string text, IEnumerable<DetectedEntity> detections,
        IDictionary<string, string> restore, IDictionary<string, int> counts)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(detections);
        ArgumentNullException.ThrowIfNull(restore);
        ArgumentNullException.ThrowIfNull(counts);

        var accepted = detections
            .Where(item => item.Start >= 0 && item.Length > 0 && item.Start + item.Length <= text.Length)
            .OrderBy(item => item.Start)
            .ToList();

        // Trim the padding and adjacent punctuation out of each span, then merge what overlaps:
        // two labels claiming the same characters would otherwise mint two placeholders for one
        // value and the second replacement would corrupt the first.
        var merged = new List<DetectedEntity>();
        foreach (var item in accepted)
        {
            var start = item.Start;
            var end = item.Start + item.Length;
            while (start < end && IsLeadingBoundary(text[start])) start++;
            while (end > start && IsTrailingBoundary(text[end - 1])) end--;
            if (start == end) continue;
            var trimmed = item with { Start = start, Length = end - start };
            if (merged.Count > 0 && trimmed.Start <= merged[^1].Start + merged[^1].Length)
            {
                var previous = merged[^1];
                var mergedEnd = Math.Max(previous.Start + previous.Length, trimmed.Start + trimmed.Length);
                merged[^1] = previous with { Length = mergedEnd - previous.Start };
            }
            else merged.Add(trimmed);
        }

        var replacements = new List<(DetectedEntity Entity, string Placeholder)>();
        foreach (var item in merged)
        {
            var label = NonAlphanumeric.Replace(item.Label.ToUpperInvariant(), "_").Trim('_');
            counts.TryGetValue(label, out var count);
            var placeholder = $"[{label}_{count + 1}]";
            counts[label] = count + 1;
            restore[placeholder] = text.Substring(item.Start, item.Length);
            replacements.Add((item, placeholder));
        }

        // Right to left, so an earlier replacement cannot shift a later span's offsets.
        var output = text;
        foreach (var replacement in replacements.OrderByDescending(item => item.Entity.Start))
            output = output.Remove(replacement.Entity.Start, replacement.Entity.Length)
                .Insert(replacement.Entity.Start, replacement.Placeholder);
        return output;
    }

    /// <summary>Put the originals back, then scrub any placeholder of ours the map cannot account for.</summary>
    /// <remarks>
    /// The scrub is scoped to labels this masker can actually mint, rather than every bracketed
    /// token. A general sweep would delete bracketed text the user typed themselves — "See
    /// [FIGURE_2] for the numbers." loses the reference, and a caller diffing the reply against the
    /// user's unmasked draft reports the deletion as a correction.
    /// </remarks>
    /// <param name="text">The reply to restore into.</param>
    /// <param name="restore">The map <see cref="MaskAsync"/> returned.</param>
    public static string Restore(string text, IReadOnlyDictionary<string, string> restore)
    {
        ArgumentNullException.ThrowIfNull(text);
        ArgumentNullException.ThrowIfNull(restore);
        // Longest key first: [EMAIL_ADDRESS_11] must not be eaten by [EMAIL_ADDRESS_1].
        foreach (var pair in restore.OrderByDescending(pair => pair.Key.Length))
            text = text.Replace(pair.Key, pair.Value, StringComparison.Ordinal);
        return MintedPlaceholders.Replace(text, string.Empty);
    }

    private static readonly Regex MintedPlaceholders = new(
        @"\[(?:" + string.Join("|", MaskerConfig.DefaultPII.Select(label =>
            Regex.Replace(label.ToUpperInvariant(), "[^A-Z0-9]+", "_").Trim('_')))
        + @")_[0-9]+\]", RegexOptions.Compiled);

    // MARK: probe

    /// <summary>The text the health probe masks.</summary>
    /// <remarks>
    /// Deliberately plain and English-neutral in shape: the point is to catch a model that detects
    /// nothing, not to measure quality.
    /// </remarks>
    public const string ProbeText =
        "Email me at jane.doe@example.com or call 415-555-0142, my IP is 192.168.4.21.";

    /// <summary>The anchors a working model has to cover.</summary>
    /// <remarks>
    /// Label-agnostic on purpose — what matters for the privacy gate is that the span gets masked at
    /// all, not which label it was filed under.
    /// </remarks>
    public static IReadOnlyList<string> ProbeAnchors { get; } = new ReadOnlyCollection<string>(
        ["jane.doe@example.com", "415-555-0142", "192.168.4.21"]);

    /// <summary>Run the probe against whatever this masker resolves to right now.</summary>
    /// <remarks>
    /// The library ships this check but never calls it. Running it is the host's job: call it once
    /// after an install reports new bytes, and treat <see cref="ModelProbeResult.Passes"/> as the
    /// pass mark. It is what <see cref="ModelRollback.VerifyAsync"/> expects to be handed.
    /// </remarks>
    /// <param name="cancellationToken">Cancels the probe; a cancelled probe reports that it did not run.</param>
    public async Task<ModelProbeResult> ProbeAsync(CancellationToken cancellationToken = default)
    {
        var masked = await MaskAsync([ProbeText], cancellationToken).ConfigureAwait(false);
        if (masked == null) return ModelProbeResult.DidNotRun;
        var covered = masked.Restore.Values;
        var found = ProbeAnchors.Where(a => covered.Any(v => v.Contains(a, StringComparison.Ordinal))).ToList();
        var missed = ProbeAnchors.Where(a => !found.Contains(a, StringComparer.Ordinal)).ToList();
        return new ModelProbeResult(Ran: true, Found: found, Missed: missed);
    }

    // MARK: inference

    private List<DetectedEntity> Detect(string text, RunOptions runOptions, CancellationToken cancellationToken)
    {
        if (string.IsNullOrWhiteSpace(text)) return [];
        var matches = Words.Matches(text).Cast<Match>().ToArray();
        var wordTokens = matches
            .Select(match => RawTokens(IsEmojiOnly(match.Value) ? "-" : match.Value))
            .ToArray();
        if (wordTokens.Sum(tokens => tokens.Length) > config.MaxInputTokens)
            throw new InvalidDataException("Privacy input exceeds the model token limit.");

        var preamble = new List<long> { 1 };
        foreach (var label in config.Labels)
        {
            preamble.Add(128001);
            preamble.AddRange(RawTokens(label).Select(id => (long)id));
        }
        preamble.Add(128002);

        // The label preamble shares the window with the text, so the text's budget is what is left.
        var capacity = pin.MaxSequenceLength - preamble.Count - 1;
        if (wordTokens.Any(tokens => tokens.Length > capacity))
            throw new InvalidDataException("A privacy input word exceeds the model window.");

        var detections = new List<DetectedEntity>();
        var first = 0;
        while (first < matches.Length)
        {
            cancellationToken.ThrowIfCancellationRequested();
            var last = first;
            var used = 0;
            while (last < matches.Length && used + wordTokens[last].Length <= capacity)
                used += wordTokens[last++].Length;
            detections.AddRange(RunWindow(matches, wordTokens, first, last, runOptions));
            if (last == matches.Length) break;
            // Overlap the next window by the maximum span width, so an entity straddling the
            // boundary is still wholly inside one window.
            first = Math.Max(first + 1, last - (pin.MaxWidth - 1));
        }
        return detections;
    }

    private IEnumerable<DetectedEntity> RunWindow(
        Match[] matches, uint[][] tokens, int first, int last, RunOptions runOptions)
    {
        var wordCount = last - first;
        var ids = new List<long> { 1 };
        var wordMask = new List<long> { 0 };
        foreach (var label in config.Labels)
        {
            ids.Add(128001); wordMask.Add(0);
            foreach (var id in RawTokens(label)) { ids.Add(id); wordMask.Add(0); }
        }
        ids.Add(128002); wordMask.Add(0);
        for (var index = first; index < last; index++)
            for (var part = 0; part < tokens[index].Length; part++)
            {
                ids.Add(tokens[index][part]);
                wordMask.Add(part == 0 ? index - first + 1 : 0);
            }
        ids.Add(2); wordMask.Add(0);

        var spanCount = wordCount * pin.MaxWidth;
        var spanIndices = new long[spanCount * 2];
        var spanMask = new long[spanCount];
        for (var word = 0; word < wordCount; word++)
            for (var width = 0; width < pin.MaxWidth; width++)
            {
                var offset = word * pin.MaxWidth + width;
                spanIndices[offset * 2] = word;
                spanIndices[offset * 2 + 1] = word + width;
                spanMask[offset] = word + width < wordCount ? 1 : 0;
            }

        var inputs = new[]
        {
            NamedOnnxValue.CreateFromTensor("input_ids", new DenseTensor<long>(ids.ToArray(), [1, ids.Count])),
            NamedOnnxValue.CreateFromTensor("attention_mask", new DenseTensor<long>(Enumerable.Repeat(1L, ids.Count).ToArray(), [1, ids.Count])),
            NamedOnnxValue.CreateFromTensor("words_mask", new DenseTensor<long>(wordMask.ToArray(), [1, ids.Count])),
            NamedOnnxValue.CreateFromTensor("text_lengths", new DenseTensor<long>(new[] { (long)wordCount }, new[] { 1, 1 })),
            NamedOnnxValue.CreateFromTensor("span_idx", new DenseTensor<long>(spanIndices, [1, spanCount, 2])),
            NamedOnnxValue.CreateFromTensor("span_mask_int64", new DenseTensor<long>(spanMask, [1, spanCount])),
        };
        using var results = session!.Run(inputs, ["logits"], runOptions);
        var logits = results[0].AsTensor<float>().ToArray();
        for (var word = 0; word < wordCount; word++)
            for (var width = 0; width < pin.MaxWidth && word + width < wordCount; width++)
                for (var label = 0; label < config.Labels.Count; label++)
                {
                    var value = logits[(word * pin.MaxWidth + width) * config.Labels.Count + label];
                    var confidence = 1f / (1f + MathF.Exp(-value));
                    if (confidence <= config.Threshold) continue;
                    var startMatch = matches[first + word];
                    var endMatch = matches[first + word + width];
                    yield return new(
                        startMatch.Index,
                        endMatch.Index + endMatch.Length - startMatch.Index,
                        config.Labels[label], confidence);
                }
    }

    private uint[] RawTokens(string text)
    {
        var encoded = tokenizer!.Encode(text);
        // Drop the tokenizer's own BOS/EOS: these ids are spliced into a sequence that supplies its own.
        return encoded.Length <= 2 ? [] : encoded.Skip(1).SkipLast(1).ToArray();
    }

    private void EnsureLoaded()
    {
        if (tokenizer != null && session != null) return;
        var modelPath = Path.Combine(modelDirectory, "model.onnx");
        var tokenizerPath = Path.Combine(modelDirectory, "tokenizer.json");
        var tokenizerConfigPath = Path.Combine(modelDirectory, "tokenizer_config.json");
        // All three, before any session exists. The weights hash alone would admit a swapped
        // tokenizer, which changes what the model sees without changing the model.
        VerifyHash(tokenizerPath, pin.Files.TokenizerSha256);
        VerifyHash(tokenizerConfigPath, pin.Files.TokenizerConfigSha256);
        VerifyHash(modelPath, pin.WeightsSha256);
        tokenizer = new Tokenizer(tokenizerPath);
        session = new InferenceSession(modelPath, new SessionOptions { IntraOpNumThreads = 2 });
        logging.Note($"privacy: model {pin.Version} ready");
    }

    private void VerifyHash(string path, string expected)
    {
        using var stream = File.OpenRead(path);
        var hash = Convert.ToHexString(SHA256.HashData(stream)).ToLowerInvariant();
        if (InstallSupport.DigestsMatch(expected, hash)) return;
        logging.Note($"privacy: model {pin.Version} hash mismatch file={Path.GetFileName(path)}");
        throw new InvalidDataException("A privacy model file does not match the compiled pin.");
    }

    private static bool IsLeadingBoundary(char value) => char.IsWhiteSpace(value)
        || value is '(' or '[' or '{' or '"' or '\'' or '“' or '‘' or '«'
            or '¿' or '¡';

    private static bool IsTrailingBoundary(char value) => char.IsWhiteSpace(value)
        || value is '.' or ',' or '?' or '!' or ';' or ':' or '…' or ')' or ']'
            or '}' or '"' or '\'' or '”' or '’' or '»';

    private static bool IsEmojiOnly(string value) => value.EnumerateRunes().All(rune =>
        !Rune.IsLetterOrDigit(rune) && !Rune.IsWhiteSpace(rune)
        && !char.IsPunctuation((char)Math.Min(rune.Value, char.MaxValue)));

    /// <inheritdoc />
    public void Dispose()
    {
        session?.Dispose();
        tokenizer?.Dispose();
        inferenceGate.Dispose();
    }
}

using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// The gate every load passes through: all three model files are hash-checked against the compiled
/// pin before a session exists, and a load that fails says so loudly enough to be recoverable.
/// </summary>
/// <remarks>
/// <para>
/// This is the check that makes the pin mean anything at run time. The installer verifies bytes on
/// the way in, but a model directory is an ordinary folder that outlives the install — it can be
/// edited, partially restored from a backup, or synced over. Verifying all three on load is what
/// stops a swapped <c>tokenizer.json</c> from changing what the model sees without changing the
/// model.
/// </para>
/// <para>
/// The failure reporting matters as much as the refusal. A model that will not load is invisible
/// otherwise: the masker fails closed, the host's gate holds every request, and the application
/// looks perfectly healthy while masking nothing. So a failed warm-up records why, and marks the
/// store unusable, which is the only signal anything downstream can act on.
/// </para>
/// <para>
/// Runs without weights: the pin is built to match the fixture, so the gate can be exercised on
/// small files. In the same non-parallel collection as <see cref="ModelStoreTests"/>, because both
/// assert on the process-wide store.
/// </para>
/// </remarks>
[Collection(nameof(ModelStoreTests))]
public sealed class ModelLoadTests : IDisposable
{
    private readonly string directory = Path.Combine(
        Path.GetTempPath(), "piimasker-load-" + Guid.NewGuid().ToString("N")[..12]);

    public ModelLoadTests()
    {
        Directory.CreateDirectory(directory);
        Write("model.onnx", "not really a model");
        Write("tokenizer.json", "{\"tokenizer\":true}");
        Write("tokenizer_config.json", "{\"config\":true}");
        ModelStore.Invalidate();
    }

    public void Dispose()
    {
        ModelStore.Invalidate();
        try { Directory.Delete(directory, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private void Write(string name, string content) =>
        File.WriteAllText(Path.Combine(directory, name), content);

    private static string Sha256(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();

    /// <summary>A pin that matches the fixture as written, so only a tampered file fails.</summary>
    private ModelPin MatchingPin() => ModelPin.Current with
    {
        WeightsSha256 = Sha256(File.ReadAllText(Path.Combine(directory, "model.onnx"))),
        Files = new ModelFileManifest(
            Sha256(File.ReadAllText(Path.Combine(directory, "tokenizer.json"))),
            Sha256(File.ReadAllText(Path.Combine(directory, "tokenizer_config.json")))),
    };

    /// <summary>
    /// Any one of the three files failing its hash refuses the load, and says so on the store.
    /// </summary>
    /// <remarks>
    /// Each file separately, because each is a different attack: swapped weights change what the
    /// model knows, a swapped tokenizer changes how text is cut up before it ever reaches the
    /// model, and the config carries the special-token ids the sequence is built from.
    /// </remarks>
    [Theory]
    [InlineData("model.onnx")]
    [InlineData("tokenizer.json")]
    [InlineData("tokenizer_config.json")]
    public async Task ATamperedModelFileIsRefusedAndRecorded(string tampered)
    {
        var pin = MatchingPin();
        Write(tampered, "tampered with");
        using var filter = new PrivacyFilter(directory, pin);

        Assert.False(await filter.WarmUpAsync());

        Assert.NotNull(filter.WarmUpFailure);
        Assert.IsType<InvalidDataException>(filter.WarmUpFailure);
        Assert.Equal(ModelStore.State.Unusable, ModelStore.CurrentState);
    }

    /// <summary>
    /// A load that gets past the hashes and then fails is reported the same way.
    /// </summary>
    /// <remarks>
    /// The hashes match here, so this is the path where the bytes are exactly what was pinned and
    /// the runtime still cannot open them — a corrupt publish, or a model built for a newer ONNX
    /// Runtime. It has to be as visible as tampering, because to the user it is the same outcome.
    /// </remarks>
    [Fact]
    public async Task AFileThatPassesItsHashAndStillWillNotLoadIsRecorded()
    {
        using var filter = new PrivacyFilter(directory, MatchingPin());

        Assert.False(await filter.WarmUpAsync());

        Assert.NotNull(filter.WarmUpFailure);
        Assert.IsNotType<InvalidDataException>(filter.WarmUpFailure);
        Assert.Equal(ModelStore.State.Unusable, ModelStore.CurrentState);
    }

    /// <summary>A masker that cannot load masks nothing rather than returning the input.</summary>
    /// <remarks>
    /// The fail-closed contract, at the one place where failing open would be silent: the caller
    /// asked for masked text and would send whatever came back.
    /// </remarks>
    [Fact]
    public async Task AMaskerThatCannotLoadReturnsNullRatherThanTheInput()
    {
        var pin = MatchingPin();
        Write("tokenizer.json", "tampered with");
        using var filter = new PrivacyFilter(directory, pin);

        Assert.Null(await filter.MaskAsync(["Call me on 415-555-0142."]));
    }

    /// <summary>Concurrent callers share one load attempt rather than racing into several.</summary>
    /// <remarks>
    /// A host warms up on a background thread and may also take a request immediately. Two loads of
    /// the same session is wasted memory at best; here it also means the failure is recorded twice
    /// and the store's state flaps.
    /// </remarks>
    [Fact]
    public async Task ConcurrentWarmUpsShareOneAttempt()
    {
        using var filter = new PrivacyFilter(directory, MatchingPin());

        var results = await Task.WhenAll(Enumerable.Range(0, 8).Select(_ => filter.WarmUpAsync()));

        Assert.All(results, result => Assert.False(result));
        Assert.Equal(ModelStore.State.Unusable, ModelStore.CurrentState);
    }

    /// <summary>The diagnostic callback is told which file failed, without being told its contents.</summary>
    /// <remarks>
    /// A host's log is the only place a support case can start from. Naming the file turns "masking
    /// stopped working" into something actionable; naming its contents would put model bytes — and,
    /// on other paths, user text — into a log that outlives the process.
    /// </remarks>
    [Fact]
    public async Task TheDiagnosticNamesTheFailingFile()
    {
        var pin = MatchingPin();
        Write("tokenizer_config.json", "tampered with");
        var notes = new System.Collections.Generic.List<string>();
        using var filter = new PrivacyFilter(
            directory, pin, logging: new MaskerLogging { Diagnostic = notes.Add });

        await filter.WarmUpAsync();

        Assert.Contains(notes, note => note.Contains("tokenizer_config.json", StringComparison.Ordinal));
        Assert.DoesNotContain(notes, note => note.Contains("tampered with", StringComparison.Ordinal));
    }
}

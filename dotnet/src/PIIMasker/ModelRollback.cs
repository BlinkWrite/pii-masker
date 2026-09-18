using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using System.Threading;
using System.Threading.Tasks;

namespace PIIMasker;

/// <summary>What a health probe concluded about an installed model.</summary>
/// <param name="Ran">Whether the probe executed at all. False means the model could not even be opened.</param>
/// <param name="Found">Anchors the model detected.</param>
/// <param name="Missed">Anchors it should have detected and did not.</param>
public sealed record ModelProbeResult(bool Ran, IReadOnlyList<string> Found, IReadOnlyList<string> Missed)
{
    /// <summary>The pass mark: the probe ran and covered at least two of the three anchors.</summary>
    /// <remarks>
    /// Two, not all three: one miss is model jitter, two is a model that is not doing its job. A
    /// version that fails this leaves the fail-closed gate open, which is worse than a crash —
    /// it leaks silently. Requiring a clean sweep instead would revert healthy models over noise.
    /// </remarks>
    public bool Passes => Ran && Found.Count >= 2;

    /// <summary>A probe that could not run at all.</summary>
    public static ModelProbeResult DidNotRun { get; } =
        new(Ran: false, Found: Array.Empty<string>(), Missed: PrivacyFilter.ProbeAnchors);
}

/// <summary>Where a rollback keeps its two records between launches.</summary>
/// <remarks>
/// The Swift target takes a <c>UserDefaults</c> and two key names. .NET has no equivalent that is
/// right on every platform a host might ship to, so the store is an interface: a Windows host can
/// back it with the registry, a macOS one with a plist, and anything else with
/// <see cref="FileRollbackStore"/>. Keeping it injectable is also what lets the tests drive the
/// launch-counter path without touching the machine.
/// </remarks>
public interface IRollbackStore
{
    /// <summary>Read a stored value, or null if it was never written.</summary>
    /// <param name="key">The record's key.</param>
    string? Read(string key);

    /// <summary>Write a value, or remove it when <paramref name="value"/> is null.</summary>
    /// <param name="key">The record's key.</param>
    /// <param name="value">The value to store, or null to remove.</param>
    void Write(string key, string? value);
}

/// <summary>A <see cref="IRollbackStore"/> backed by one small JSON file.</summary>
/// <remarks>
/// Written through a temporary and renamed, because a torn write here pairs a version with the
/// wrong attempt count — which is how a healthy model gets reverted, or a broken one kept.
/// </remarks>
public sealed class FileRollbackStore : IRollbackStore
{
    private readonly string path;
    // A plain object rather than System.Threading.Lock: that type is .NET 9+, and this library
    // targets net8.0 so a host on the LTS runtime can consume it.
    private readonly object gate = new();

    /// <param name="path">The JSON file to keep the records in.</param>
    public FileRollbackStore(string path)
    {
        ArgumentException.ThrowIfNullOrEmpty(path);
        this.path = path;
    }

    /// <inheritdoc />
    public string? Read(string key)
    {
        lock (gate)
        {
            var all = Load();
            return all.TryGetValue(key, out var value) ? value : null;
        }
    }

    /// <inheritdoc />
    public void Write(string key, string? value)
    {
        lock (gate)
        {
            var all = Load();
            if (value == null) all.Remove(key); else all[key] = value;
            var directory = Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory)) Directory.CreateDirectory(directory);
            var temp = path + ".tmp";
            try
            {
                File.WriteAllText(temp, JsonSerializer.Serialize(all));
                File.Move(temp, path, overwrite: true);
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException)
            {
                try { File.Delete(temp); }
                catch (IOException) { }
                catch (UnauthorizedAccessException) { }
                throw;
            }
        }
    }

    private Dictionary<string, string> Load()
    {
        try
        {
            return File.Exists(path)
                ? JsonSerializer.Deserialize<Dictionary<string, string>>(File.ReadAllText(path)) ?? []
                : [];
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException or JsonException)
        {
            // An unreadable record is treated as no record. The cost is one unproved model losing
            // its probation, which the launch counter then catches; the alternative — throwing on
            // the launch path — would make a corrupt settings file stop the host from starting.
            return [];
        }
    }
}

/// <summary>Probation for a freshly-installed model version, and the revert back to the one it replaced.</summary>
/// <remarks>
/// <para>
/// The payload is already on disk — superseded version directories are deliberately never pruned —
/// so the record is just the version pair, and reverting is one atomic pointer flip.
/// </para>
/// <para>
/// A bad model does not crash. Either the loader fails and the fail-closed gate holds every request,
/// so the host looks alive while doing nothing; or, worse, the model loads and detects nothing and
/// the gate stops closing at all. So the health signal is not "did we survive" but "does it still
/// find PII". The one failure that <i>can</i> crash — the runtime dying on a malformed graph — is
/// caught by the launch counter instead, since nothing in-process gets to report a verdict there.
/// </para>
/// </remarks>
public sealed class ModelRollback
{
    /// <summary>
    /// Launches an unresolved probation gets before the version is reverted unprobed. Two, not one:
    /// an unrelated crash during launch would otherwise blocklist a model that was never at fault.
    /// </summary>
    public const int MaxLaunchAttempts = 2;

    /// <summary>The two record keys this type owns, so a host can keep its own namespace.</summary>
    /// <param name="Probation">Key for the in-flight probation record.</param>
    /// <param name="FailedVersions">Key for the refused-version list.</param>
    public sealed record SettingsKeys(string Probation, string FailedVersions)
    {
        /// <summary>
        /// The library's own key names, used when a host supplies none of its own.
        /// </summary>
        /// <remarks>
        /// The Swift target's <c>SettingsKeys.default</c> holds these same two strings and has to
        /// keep agreeing with them: both targets can be pointed at one store, and a rollback record
        /// written by either has to be the record the other reads.
        ///
        /// They are persisted keys, so changing them once something relies on them abandons what
        /// was written under the old names — a model part-way through probation stops being
        /// watched, and versions already known to fail get offered again. Nothing relies on them
        /// yet, which is why they could still be renamed with the repository: the one shipping
        /// consumer passes its own pair, which is what this fallback exists for.
        /// </remarks>
        public static SettingsKeys Default { get; } = new(
            "com.github.pii-masker.modelUpdate.probation",
            "com.github.pii-masker.modelUpdate.failedVersions");
    }

    /// <summary>What <see cref="ResolveAtLaunch"/> decided.</summary>
    public enum LaunchOutcome
    {
        /// <summary>Nothing on probation.</summary>
        None,
        /// <summary>A version is on probation and has launches left.</summary>
        Proving,
        /// <summary>Reverted at launch without probing — the previous launches died before reaching a verdict.</summary>
        RolledBack,
    }

    /// <summary>What <see cref="VerifyAsync"/> decided.</summary>
    public enum Verdict
    {
        /// <summary>Nothing on probation.</summary>
        Idle,
        /// <summary>The model found what it was meant to find.</summary>
        Verified,
        /// <summary>The model failed its probe and the previous version was restored.</summary>
        Reverted,
        /// <summary>
        /// The probe could not run at all. Deliberately not a failure: the model may be fine and the
        /// machine merely out of memory. Left armed so the next launch tries again — and if it never
        /// gets a verdict, <see cref="ResolveAtLaunch"/> reverts it once the launches run out.
        /// </summary>
        Inconclusive,
    }

    private sealed record Record(
        [property: JsonPropertyName("new")] string New,
        [property: JsonPropertyName("previous")] string Previous,
        [property: JsonPropertyName("attempts")] int Attempts);

    private readonly string installRoot;
    private readonly IRollbackStore store;
    private readonly SettingsKeys keys;

    /// <param name="installRoot">The model store this rollback governs.</param>
    /// <param name="store">Where the records live between launches.</param>
    /// <param name="keys">Record key names; defaults to the library's own.</param>
    public ModelRollback(string installRoot, IRollbackStore store, SettingsKeys? keys = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(installRoot);
        ArgumentNullException.ThrowIfNull(store);
        this.installRoot = installRoot;
        this.store = store;
        this.keys = keys ?? SettingsKeys.Default;
    }

    private Record? Current
    {
        get
        {
            var raw = store.Read(keys.Probation);
            if (string.IsNullOrEmpty(raw)) return null;
            try { return JsonSerializer.Deserialize<Record>(raw); }
            catch (JsonException) { return null; }
        }
    }

    private void Write(Record record) =>
        store.Write(keys.Probation, JsonSerializer.Serialize(record));

    /// <summary>The version being proved and the one it replaced, or null when nothing is armed.</summary>
    public (string New, string Previous)? Probation =>
        Current is { } r ? (r.New, r.Previous) : null;

    /// <summary>Whether a version is currently on probation.</summary>
    public bool IsPending => Current != null;

    /// <summary>Versions a probe has refused. The installer skips these while a working model exists.</summary>
    public IReadOnlySet<string> BlockedVersions
    {
        get
        {
            var raw = store.Read(keys.FailedVersions);
            if (string.IsNullOrEmpty(raw)) return new HashSet<string>(StringComparer.Ordinal);
            try
            {
                return new HashSet<string>(
                    JsonSerializer.Deserialize<List<string>>(raw) ?? [], StringComparer.Ordinal);
            }
            catch (JsonException) { return new HashSet<string>(StringComparer.Ordinal); }
        }
    }

    /// <summary>Put a freshly-installed version on probation.</summary>
    /// <remarks>
    /// Only armed when there is something to go back to — either the previous version's bytes are
    /// already on disk, or it is a known pin that can be fetched again. A first install and a
    /// same-version repair have no previous version at all, so a failure there is a reinstall, not
    /// a rollback.
    /// </remarks>
    /// <param name="newVersion">The version just installed.</param>
    /// <param name="previousVersion">The version it replaced.</param>
    public void Arm(string newVersion, string previousVersion)
    {
        if (string.Equals(newVersion, previousVersion, StringComparison.Ordinal)) return;
        if (!CanRevertTo(previousVersion)) return;
        Write(new Record(newVersion, previousVersion, Attempts: 0));
    }

    private bool CanRevertTo(string version) =>
        ModelInstaller.IsCompleteModelDir(Path.Combine(installRoot, version))
        || ModelPin.Known.Any(p => string.Equals(p.Version, version, StringComparison.Ordinal));

    /// <summary>Clear the probation record — the model proved itself.</summary>
    public void Confirm() => store.Write(keys.Probation, null);

    /// <summary>Refuse a version from now on.</summary>
    /// <param name="version">The version to refuse.</param>
    public void Block(string version)
    {
        var blocked = new List<string>(BlockedVersions);
        if (blocked.Contains(version, StringComparer.Ordinal)) return;
        blocked.Add(version);
        store.Write(keys.FailedVersions, JsonSerializer.Serialize(blocked));
    }

    /// <summary>Forget every refusal.</summary>
    /// <remarks>
    /// A user asking for a reinstall outranks our verdict — they may know the failure was
    /// environmental (a half-written disk, a killed process), and without this the only pinned
    /// version stays refused forever with no way to say "try again".
    /// </remarks>
    public void ClearBlocklist() => store.Write(keys.FailedVersions, null);

    /// <summary>Spend one of the probation's launches, or revert it if they have run out.</summary>
    /// <remarks>
    /// Cheap and synchronous, for the top of startup: no model load, no inference — just "has this
    /// probation already burned its launches?". The probe itself runs later, off the launch path.
    /// </remarks>
    public LaunchOutcome ResolveAtLaunch()
    {
        if (Current is not { } record) return LaunchOutcome.None;
        if (record.Attempts >= MaxLaunchAttempts)
            return Revert() ? LaunchOutcome.RolledBack : LaunchOutcome.None;
        Write(record with { Attempts = record.Attempts + 1 });
        return LaunchOutcome.Proving;
    }

    /// <summary>Probe the model on probation and apply the verdict.</summary>
    /// <remarks>
    /// <para>
    /// The verdict is applied only if the record still names the version this started on — an
    /// install landing mid-probe moves the target, and applying a stale verdict would blocklist the
    /// wrong version.
    /// </para>
    /// <para>
    /// The probe is a required argument here, where the Swift target defaults it to the filter's own.
    /// The inference half has not been ported yet, and defaulting it to something that cannot detect
    /// anything would make every model fail its probe and revert — the exact damage this type exists
    /// to prevent. It becomes optional when the filter lands.
    /// </para>
    /// </remarks>
    /// <param name="probe">Loads the model at the given directory and reports what it detected.</param>
    /// <param name="cancellationToken">Abandons the probe; the probation is left armed.</param>
    public async Task<Verdict> VerifyAsync(
        Func<string, CancellationToken, Task<ModelProbeResult>> probe,
        CancellationToken cancellationToken = default)
    {
        ArgumentNullException.ThrowIfNull(probe);
        if (Probation is not { } started) return Verdict.Idle;

        // Pinned to this store rather than resolved: a verdict formed against a different install
        // root would revert and blocklist a version it never actually loaded.
        var directory = ModelStore.ResolvePromoted(installRoot);
        var result = directory == null
            ? ModelProbeResult.DidNotRun
            : await probe(directory, cancellationToken).ConfigureAwait(false);

        if (Probation?.New != started.New) return Verdict.Idle;

        if (result.Passes)
        {
            Confirm();
            return Verdict.Verified;
        }

        // A probe that never ran is only a verdict about the model if the files were all there to
        // begin with: then they were opened and were not a model, which is this version,
        // definitively. A store that is missing files or was repointed mid-probe says nothing about
        // it, and blocklisting on that would pin the machine to an older version over a one-off
        // hiccup.
        if (!result.Ran && !ModelInstaller.IsCompleteModelDir(directory)) return Verdict.Inconclusive;

        return Revert() ? Verdict.Reverted : Verdict.Idle;
    }

    /// <summary>
    /// The pinned model a revert would have to download because its bytes are not on disk, or null
    /// when the flip needs no network.
    /// </summary>
    public ModelPin? PendingRevertDownload
    {
        get
        {
            if (Probation is not { } probation) return null;
            if (ModelInstaller.IsCompleteModelDir(Path.Combine(installRoot, probation.Previous))) return null;
            return ModelPin.Known.FirstOrDefault(
                p => string.Equals(p.Version, probation.Previous, StringComparison.Ordinal));
        }
    }

    /// <summary>Flip the pointer back and refuse the version that failed. No network.</summary>
    /// <remarks>
    /// Verified before the flip — a previous version that has since been deleted or truncated would
    /// take a working-but-suspect model and replace it with nothing at all, which is strictly worse
    /// than what we are recovering from. When those bytes are gone but the pin list still knows the
    /// version, the probation record is LEFT ARMED so <see cref="RevertInstallingIfNeededAsync"/>
    /// can fetch it; when nothing can bring it back, probation is cleared rather than retried forever.
    /// </remarks>
    public bool Revert()
    {
        if (Probation is not { } probation) return false;
        if (!ModelInstaller.IsCompleteModelDir(Path.Combine(installRoot, probation.Previous)))
        {
            if (PendingRevertDownload != null) return false;
            Confirm();
            return false;
        }
        try
        {
            ModelInstaller.PointCurrent(probation.Previous, installRoot);
        }
        catch (ModelInstaller.InstallException)
        {
            return false;
        }
        FinishRevert(probation.New);
        return true;
    }

    /// <summary>Revert, downloading the previous version first if its bytes are no longer on disk.</summary>
    /// <remarks>
    /// The flip path is the normal one and costs no network. The fetch path exists because a pin
    /// list can name a version this machine never installed.
    /// </remarks>
    /// <param name="cancellationToken">Abandons the download; the probation is left armed.</param>
    public async Task<bool> RevertInstallingIfNeededAsync(CancellationToken cancellationToken = default)
    {
        if (Revert()) return true;
        if (Probation is not { } probation || PendingRevertDownload is not { } pin) return false;
        try
        {
            await new ModelInstaller(installRoot, pin)
                .EnsureLatestAsync(force: true, cancellationToken: cancellationToken)
                .ConfigureAwait(false);
        }
        catch (ModelInstaller.InstallException)
        {
            return false;
        }
        FinishRevert(probation.New);
        return true;
    }

    private void FinishRevert(string failed)
    {
        Block(failed);
        Confirm();
        ModelStore.Invalidate();
    }
}

using System;
using System.Collections.Generic;
using System.Linq;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// Probation, the launch counter, and the revert. The failure this guards against is silent — a
/// model that loads fine and quietly stops detecting PII — so every branch is driven with an
/// injected probe rather than inferred from a real one.
/// </summary>
public sealed class ModelRollbackTests : IDisposable
{
    private readonly string scratch = Path.Combine(
        Path.GetTempPath(), "piimasker-rollback-" + Guid.NewGuid().ToString("N"));

    public ModelRollbackTests() => Directory.CreateDirectory(scratch);

    public void Dispose()
    {
        try { Directory.Delete(scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private string InstallRoot => scratch;

    /// <summary>An in-memory store, so a test never touches the machine's real settings.</summary>
    private sealed class MemoryStore : IRollbackStore
    {
        private readonly Dictionary<string, string> values = new(StringComparer.Ordinal);
        public string? Read(string key) => values.TryGetValue(key, out var v) ? v : null;
        public void Write(string key, string? value)
        {
            if (value == null) values.Remove(key); else values[key] = value;
        }
    }

    private ModelRollback NewRollback(IRollbackStore? store = null) =>
        new(InstallRoot, store ?? new MemoryStore());

    private void InstallVersion(string version)
    {
        var dir = Path.Combine(InstallRoot, version);
        Directory.CreateDirectory(dir);
        foreach (var name in ModelInstaller.RequiredModelFiles)
            File.WriteAllText(Path.Combine(dir, name), version);
        ModelInstaller.PointCurrent(version, InstallRoot);
    }

    /// <summary>
    /// Derives <c>Found</c> from the real anchor list rather than taking it separately, so a probe
    /// result here cannot describe a state the production probe could never produce — which is how
    /// a "healthy" fixture ends up failing the two-of-three pass mark.
    /// </summary>
    private static Func<string, CancellationToken, Task<ModelProbeResult>> Probe(
        bool ran, params string[] missed) =>
        (_, _) => Task.FromResult(new ModelProbeResult(
            ran,
            PrivacyFilter.ProbeAnchors.Where(a => !missed.Contains(a, StringComparer.Ordinal)).ToList(),
            missed));

    private static readonly Func<string, CancellationToken, Task<ModelProbeResult>> Healthy =
        Probe(ran: true);

    private static readonly Func<string, CancellationToken, Task<ModelProbeResult>> DetectsNothing =
        Probe(ran: true, [.. PrivacyFilter.ProbeAnchors]);

    [Fact]
    public void ArmingNeedsSomethingToGoBackTo()
    {
        InstallVersion("2026.08.1");
        var rollback = NewRollback();

        // No previous version on disk and not a known pin: a failure here is a reinstall, not a
        // rollback, and arming would promise a revert that cannot happen.
        rollback.Arm("2026.09.1", "never-published");
        Assert.False(rollback.IsPending);

        // A same-version repair has no previous version at all.
        rollback.Arm("2026.08.1", "2026.08.1");
        Assert.False(rollback.IsPending);

        // The previous version's bytes are on disk, so the revert is a pointer flip.
        rollback.Arm("2026.09.1", "2026.08.1");
        Assert.True(rollback.IsPending);
        Assert.Equal(("2026.09.1", "2026.08.1"), rollback.Probation);
    }

    /// <summary>
    /// Two launches, not one: an unrelated crash during launch would otherwise blocklist a model
    /// that was never at fault.
    /// </summary>
    [Fact]
    public void AProbationSurvivesOneUnrelatedCrashAndIsRevertedAfterTheSecond()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2026.08.1");

        Assert.Equal(ModelRollback.LaunchOutcome.Proving, rollback.ResolveAtLaunch());
        Assert.Equal(ModelRollback.LaunchOutcome.Proving, rollback.ResolveAtLaunch());
        Assert.Equal(ModelRollback.LaunchOutcome.RolledBack, rollback.ResolveAtLaunch());

        Assert.Equal("2026.08.1", new ModelInstaller(InstallRoot).InstalledVersion());
        Assert.Contains("2026.09.1", rollback.BlockedVersions);
        Assert.False(rollback.IsPending);
    }

    [Fact]
    public async Task AHealthyModelIsConfirmedAndNothingIsBlocked()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2026.08.1");

        var verdict = await rollback.VerifyAsync(Healthy);

        Assert.Equal(ModelRollback.Verdict.Verified, verdict);
        Assert.False(rollback.IsPending);
        Assert.Empty(rollback.BlockedVersions);
        Assert.Equal("2026.09.1", new ModelInstaller(InstallRoot).InstalledVersion());
    }

    /// <summary>
    /// The failure the whole mechanism exists for: the model loads perfectly and detects nothing, so
    /// the fail-closed gate stops closing and unmasked text starts flowing.
    /// </summary>
    [Fact]
    public async Task AModelThatLoadsButDetectsNothingIsRevertedAndRefused()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2026.08.1");

        var verdict = await rollback.VerifyAsync(DetectsNothing);

        Assert.Equal(ModelRollback.Verdict.Reverted, verdict);
        Assert.Equal("2026.08.1", new ModelInstaller(InstallRoot).InstalledVersion());
        Assert.Contains("2026.09.1", rollback.BlockedVersions);
        Assert.False(rollback.IsPending);
    }

    /// <summary>
    /// A probe that could not run against a complete directory IS a verdict: the files were all
    /// there, they were opened, and they were not a model.
    /// </summary>
    [Fact]
    public async Task AProbeThatCannotRunAgainstACompleteStoreIsAVerdict()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2026.08.1");

        var verdict = await rollback.VerifyAsync(Probe(ran: false));

        Assert.Equal(ModelRollback.Verdict.Reverted, verdict);
        Assert.Contains("2026.09.1", rollback.BlockedVersions);
    }

    /// <summary>
    /// A probe that could not run against an INCOMPLETE store says nothing about the model.
    /// Blocklisting there would pin the machine to an older version over a one-off hiccup.
    /// </summary>
    [Fact]
    public async Task AProbeThatCannotRunAgainstAnIncompleteStoreIsInconclusive()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2026.08.1");
        File.Delete(Path.Combine(InstallRoot, "2026.09.1", "tokenizer.json"));

        var verdict = await rollback.VerifyAsync(Probe(ran: false));

        Assert.Equal(ModelRollback.Verdict.Inconclusive, verdict);
        Assert.Empty(rollback.BlockedVersions);
        // Left armed, so the next launch tries again and the counter eventually resolves it.
        Assert.True(rollback.IsPending);
    }

    /// <summary>
    /// An install landing mid-probe moves the target. Applying the verdict then would blocklist a
    /// version the probe never looked at.
    /// </summary>
    [Fact]
    public async Task AVerdictIsDiscardedWhenAnInstallMovesTheTargetMidProbe()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2026.08.1");

        var verdict = await rollback.VerifyAsync((_, _) =>
        {
            // A newer install re-arms probation while the probe is out.
            rollback.Arm("2026.10.1", "2026.09.1");
            return Task.FromResult(new ModelProbeResult(true, [], ["email address"]));
        });

        Assert.Equal(ModelRollback.Verdict.Idle, verdict);
        Assert.Empty(rollback.BlockedVersions);
        Assert.Equal(("2026.10.1", "2026.09.1"), rollback.Probation);
    }

    /// <summary>
    /// Reverting to bytes that are gone would replace a working-but-suspect model with nothing at
    /// all — strictly worse than what is being recovered from. When the missing version is still a
    /// known pin it can be fetched, so probation is LEFT ARMED for
    /// <see cref="ModelRollback.RevertInstallingIfNeededAsync"/> rather than abandoned.
    /// </summary>
    [Fact]
    public void ARevertWaitsForADownloadWhenTheMissingVersionIsStillAKnownPin()
    {
        var known = ModelPin.Current.Version;
        InstallVersion(known);
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", known);
        Directory.Delete(Path.Combine(InstallRoot, known), recursive: true);

        var reverted = rollback.Revert();

        Assert.False(reverted);
        Assert.Equal("2026.09.1", new ModelInstaller(InstallRoot).InstalledVersion());
        Assert.NotNull(rollback.PendingRevertDownload);
        Assert.True(rollback.IsPending);
        // Nothing is blocked yet: the version on probation has not been judged, only deferred.
        Assert.Empty(rollback.BlockedVersions);
    }

    /// <summary>
    /// When the previous version's bytes are gone AND no pin can bring it back, probation is cleared
    /// rather than retried forever — there is no recovery to wait for.
    /// </summary>
    [Fact]
    public void ARevertIsAbandonedWhenNothingCanBringThePreviousVersionBack()
    {
        InstallVersion("2020.01.1");
        InstallVersion("2026.09.1");
        var rollback = NewRollback();
        rollback.Arm("2026.09.1", "2020.01.1");
        Directory.Delete(Path.Combine(InstallRoot, "2020.01.1"), recursive: true);

        var reverted = rollback.Revert();

        Assert.False(reverted);
        Assert.Null(rollback.PendingRevertDownload);
        Assert.False(rollback.IsPending);
        Assert.Equal("2026.09.1", new ModelInstaller(InstallRoot).InstalledVersion());
    }

    [Fact]
    public void ClearingTheBlocklistLetsARefusedVersionBeTriedAgain()
    {
        var rollback = NewRollback();
        rollback.Block("2026.09.1");
        Assert.Contains("2026.09.1", rollback.BlockedVersions);

        rollback.ClearBlocklist();

        Assert.Empty(rollback.BlockedVersions);
    }

    [Fact]
    public void BlockingIsIdempotent()
    {
        var rollback = NewRollback();
        rollback.Block("2026.09.1");
        rollback.Block("2026.09.1");

        Assert.Single(rollback.BlockedVersions);
    }

    [Fact]
    public async Task NothingArmedMeansNothingToVerify()
    {
        Assert.Equal(ModelRollback.Verdict.Idle, await NewRollback().VerifyAsync(DetectsNothing));
        Assert.Equal(ModelRollback.LaunchOutcome.None, NewRollback().ResolveAtLaunch());
    }

    /// <summary>
    /// The file-backed store has to survive a process boundary, because the launch counter is
    /// meaningless if the count resets on every start.
    /// </summary>
    [Fact]
    public void TheFileStoreSurvivesANewInstanceAndRoundTripsTheAttemptCount()
    {
        InstallVersion("2026.08.1");
        InstallVersion("2026.09.1");
        var path = Path.Combine(scratch, "rollback.json");

        var first = new ModelRollback(InstallRoot, new FileRollbackStore(path));
        first.Arm("2026.09.1", "2026.08.1");
        Assert.Equal(ModelRollback.LaunchOutcome.Proving, first.ResolveAtLaunch());

        // A second instance reading the same file is the next launch.
        var second = new ModelRollback(InstallRoot, new FileRollbackStore(path));
        Assert.Equal(("2026.09.1", "2026.08.1"), second.Probation);
        Assert.Equal(ModelRollback.LaunchOutcome.Proving, second.ResolveAtLaunch());
        Assert.Equal(ModelRollback.LaunchOutcome.RolledBack, second.ResolveAtLaunch());
    }

    /// <summary>
    /// A corrupt record must not stop the host starting. The cost is one unproved model losing its
    /// probation, which the launch counter catches; throwing on the launch path would be worse.
    /// </summary>
    [Fact]
    public void ACorruptRecordReadsAsNoRecordRatherThanThrowing()
    {
        var path = Path.Combine(scratch, "rollback.json");
        File.WriteAllText(path, "{ this is not json");
        var rollback = new ModelRollback(InstallRoot, new FileRollbackStore(path));

        Assert.False(rollback.IsPending);
        Assert.Empty(rollback.BlockedVersions);
        Assert.Equal(ModelRollback.LaunchOutcome.None, rollback.ResolveAtLaunch());
    }
}

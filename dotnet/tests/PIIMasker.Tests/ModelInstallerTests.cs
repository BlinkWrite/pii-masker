using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// The whole install pipeline over <c>file://</c>: download, verify, unpack, promote, re-run,
/// reject. No network and no weights, so this runs anywhere — the same contract the Swift
/// <c>ModelInstallerTests</c> covers.
/// </summary>
public sealed class ModelInstallerTests : IDisposable
{
    private readonly string scratch = Path.Combine(
        Path.GetTempPath(), "piimasker-tests-" + Guid.NewGuid().ToString("N"));

    public ModelInstallerTests() => Directory.CreateDirectory(scratch);

    public void Dispose()
    {
        try { Directory.Delete(scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private string InstallRoot => Path.Combine(scratch, "store");

    /// <summary>A real gzipped tar holding the three required files, so the unpack is genuinely exercised.</summary>
    private string WriteArchive(string name, IReadOnlyDictionary<string, string> entries)
    {
        var path = Path.Combine(scratch, name);
        using var file = File.Create(path);
        using var gzip = new GZipStream(file, CompressionLevel.SmallestSize);
        using var tar = new TarWriter(gzip, TarEntryFormat.Pax);
        foreach (var (entryName, content) in entries)
        {
            var bytes = Encoding.UTF8.GetBytes(content);
            var entry = new PaxTarEntry(TarEntryType.RegularFile, entryName)
            {
                DataStream = new MemoryStream(bytes),
            };
            tar.WriteEntry(entry);
        }
        return path;
    }

    private static Dictionary<string, string> CompleteModel(string weights = "the weights") =>
        new Dictionary<string, string>
        {
            ["model.onnx"] = weights,
            ["tokenizer.json"] = "{\"tokenizer\":true}",
            ["tokenizer_config.json"] = "{\"config\":true}",
        };

    private static string Sha256OfFile(string path)
    {
        using var stream = File.OpenRead(path);
        using var sha = SHA256.Create();
        return Convert.ToHexString(sha.ComputeHash(stream)).ToLowerInvariant();
    }

    private static string Sha256OfText(string text) =>
        Convert.ToHexString(SHA256.HashData(Encoding.UTF8.GetBytes(text))).ToLowerInvariant();

    private static ModelPin PinFor(string archivePath, string weights, string version = "2026.08.1") =>
        ModelPin.Current with
        {
            Version = version,
            SourceUri = new Uri(archivePath),
            ArchiveSha256 = Sha256OfFile(archivePath),
            WeightsSha256 = Sha256OfText(weights),
            Bytes = new FileInfo(archivePath).Length,
        };

    private ModelInstaller InstallerFor(ModelPin pin) => new(InstallRoot, pin);

    [Fact]
    public async Task InstallsVerifiesAndPromotes()
    {
        var archive = WriteArchive("good.tar.gz", CompleteModel());
        var installer = InstallerFor(PinFor(archive, "the weights"));

        var outcome = await installer.EnsureLatestAsync();

        Assert.True(outcome.Installed);
        Assert.True(ModelInstaller.IsCompleteModelDir(outcome.Directory));
        Assert.Equal("2026.08.1", installer.InstalledVersion());
        Assert.Equal(outcome.Directory, installer.CurrentModelDir);
    }

    /// <summary>
    /// The second run must write nothing. A host keys its post-install work — a reload, a probe, a
    /// cache drop — on <see cref="ModelInstaller.Outcome.Installed"/>, so reporting true here would
    /// fire all of it on every launch.
    /// </summary>
    [Fact]
    public async Task ASecondRunIsIdempotentAndReportsNothingInstalled()
    {
        var archive = WriteArchive("good.tar.gz", CompleteModel());
        var pin = PinFor(archive, "the weights");

        var first = await InstallerFor(pin).EnsureLatestAsync();
        var second = await InstallerFor(pin).EnsureLatestAsync();

        Assert.True(first.Installed);
        Assert.False(second.Installed);
        Assert.Equal(first.Directory, second.Directory);
    }

    [Fact]
    public async Task ForceReinstallsAnAlreadyInstalledVersion()
    {
        var archive = WriteArchive("good.tar.gz", CompleteModel());
        var pin = PinFor(archive, "the weights");
        await InstallerFor(pin).EnsureLatestAsync();

        var forced = await InstallerFor(pin).EnsureLatestAsync(force: true);

        Assert.True(forced.Installed);
        Assert.True(ModelInstaller.IsCompleteModelDir(forced.Directory));
    }

    /// <summary>A tampered archive must never reach the store, and must leave no promotion behind.</summary>
    [Fact]
    public async Task AnArchiveWhoseHashIsWrongIsRejectedAndNothingIsPromoted()
    {
        var archive = WriteArchive("tampered.tar.gz", CompleteModel());
        var pin = PinFor(archive, "the weights") with
        {
            ArchiveSha256 = new string('a', 64),
        };
        var installer = InstallerFor(pin);

        var error = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.ChecksumMismatch, error.Kind);
        Assert.Null(installer.CurrentModelDir);
    }

    /// <summary>
    /// The archive hash is packer-dependent, so it can only attest to one publisher's tarball. The
    /// weights hash is the reproducible number — a well-packed archive of the WRONG weights passes
    /// the first check and must still be refused before anything is promoted.
    /// </summary>
    [Fact]
    public async Task AWellPackedArchiveOfTheWrongWeightsIsRejectedBeforePromotion()
    {
        var archive = WriteArchive("swapped.tar.gz", CompleteModel("substituted weights"));
        var pin = PinFor(archive, "substituted weights") with
        {
            WeightsSha256 = Sha256OfText("the weights the pin expects"),
        };
        var installer = InstallerFor(pin);

        var error = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.ChecksumMismatch, error.Kind);
        Assert.Null(installer.CurrentModelDir);
    }

    [Fact]
    public async Task AnArchiveMissingARequiredFileIsRejected()
    {
        var incomplete = new Dictionary<string, string>
        {
            ["model.onnx"] = "the weights",
            ["tokenizer.json"] = "{}",
        };
        var archive = WriteArchive("incomplete.tar.gz", incomplete);
        var installer = InstallerFor(PinFor(archive, "the weights"));

        var error = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.UnpackFailed, error.Kind);
        Assert.Contains("tokenizer_config.json", error.Message, StringComparison.Ordinal);
        Assert.Null(installer.CurrentModelDir);
    }

    /// <summary>
    /// An upgrade keeps the superseded version on disk. That is what makes rollback a pointer flip
    /// rather than a re-download, so deleting it here would quietly remove the recovery path.
    /// </summary>
    [Fact]
    public async Task AnUpgradePromotesTheNewVersionAndKeepsTheOldOneOnDisk()
    {
        var oldArchive = WriteArchive("old.tar.gz", CompleteModel("old weights"));
        var newArchive = WriteArchive("new.tar.gz", CompleteModel("new weights"));
        await InstallerFor(PinFor(oldArchive, "old weights", "2026.08.1")).EnsureLatestAsync();

        var upgraded = InstallerFor(PinFor(newArchive, "new weights", "2026.09.1"));
        var outcome = await upgraded.EnsureLatestAsync();

        Assert.True(outcome.Installed);
        Assert.Equal("2026.09.1", upgraded.InstalledVersion());
        Assert.True(Directory.Exists(Path.Combine(InstallRoot, "2026.08.1")));
        Assert.Equal("new weights", File.ReadAllText(Path.Combine(outcome.Directory, "model.onnx")));
    }

    /// <summary>
    /// A refused version is skipped only while a usable model is already installed. With nothing on
    /// disk there is no working model to protect, and honouring the skip would leave the host
    /// permanently without one.
    /// </summary>
    [Fact]
    public async Task ARefusedVersionIsSkippedOnlyWhenAWorkingModelIsAlreadyInstalled()
    {
        var archive = WriteArchive("good.tar.gz", CompleteModel());
        var pin = PinFor(archive, "the weights");
        var refused = new HashSet<string> { pin.Version };

        // Nothing installed: the skip must NOT apply, or the host never gets a model at all.
        var fresh = await InstallerFor(pin).EnsureLatestAsync(skipping: refused);
        Assert.True(fresh.Installed);

        // Now one is installed, so the skip protects it.
        var later = await InstallerFor(pin).EnsureLatestAsync(skipping: refused);
        Assert.False(later.Installed);
    }

    [Theory]
    [InlineData("../escape")]
    [InlineData("with/slash")]
    [InlineData("..")]
    [InlineData("NUL")]
    public async Task AnUnsafeVersionIsRefusedBeforeAnythingTouchesTheDisk(string version)
    {
        var archive = WriteArchive("good.tar.gz", CompleteModel());
        var pin = PinFor(archive, "the weights") with { Version = version };

        var error = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => InstallerFor(pin).EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.InvalidPin, error.Kind);
    }

    [Theory]
    [InlineData(0L)]
    [InlineData(-1L)]
    [InlineData(ModelInstaller.MaxPlausibleArchiveBytes)]
    public async Task AnImplausibleArchiveSizeIsRefused(long bytes)
    {
        var archive = WriteArchive("good.tar.gz", CompleteModel());
        var pin = PinFor(archive, "the weights") with { Bytes = bytes };

        var error = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => InstallerFor(pin).EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.InvalidPin, error.Kind);
    }

    [Fact]
    public void PromotingAnUnsafeVersionIsRefused()
    {
        Directory.CreateDirectory(InstallRoot);

        var error = Assert.Throws<ModelInstaller.InstallException>(
            () => ModelInstaller.PointCurrent("../escape", InstallRoot));

        Assert.Equal(ModelInstaller.FailureKind.StoreUnwritable, error.Kind);
        Assert.False(File.Exists(Path.Combine(InstallRoot, "current")));
    }

    /// <summary>
    /// A pointer naming something outside the root must not resolve. The file is on disk, so anything
    /// able to write it could otherwise aim the loader — or a later recursive delete — anywhere.
    /// </summary>
    [Fact]
    public void APromotionPointerNamingAnUnsafeComponentDoesNotResolve()
    {
        Directory.CreateDirectory(InstallRoot);
        File.WriteAllText(Path.Combine(InstallRoot, "current"), "../../elsewhere");

        Assert.Null(ModelStore.ResolvePromoted(InstallRoot));
    }

    /// <summary>
    /// A kill mid-download leaves a ~140 MB partial that nothing else cleans up. The sweep is
    /// age-gated because an install root can be shared, and reaping a live sibling's in-flight file
    /// breaks an install that is going fine.
    /// </summary>
    [Fact]
    public async Task AbandonedPartialsAreSweptButFreshOnesAreLeftAlone()
    {
        Directory.CreateDirectory(InstallRoot);
        var abandoned = Path.Combine(InstallRoot, ".download-2026.08.1-old.tar.gz");
        var inFlight = Path.Combine(InstallRoot, ".download-2026.08.1-live.tar.gz");
        File.WriteAllText(abandoned, "stale");
        File.WriteAllText(inFlight, "a sibling process is using this");
        File.SetLastWriteTimeUtc(abandoned, DateTime.UtcNow.AddHours(-1));

        var archive = WriteArchive("good.tar.gz", CompleteModel());
        await InstallerFor(PinFor(archive, "the weights")).EnsureLatestAsync();

        Assert.False(File.Exists(abandoned));
        Assert.True(File.Exists(inFlight));
    }

    [Fact]
    public void TheDiskBudgetCoversTheArchiveAndItsUnpackWithoutOverflowing()
    {
        Assert.Equal(0, ModelInstaller.RequiredFreeBytes(0));
        Assert.Equal(0, ModelInstaller.UnpackFreeBytes(0));
        Assert.Equal(250, ModelInstaller.RequiredFreeBytes(100));
        Assert.Equal(150, ModelInstaller.UnpackFreeBytes(100));
        // A caller-built pin can carry a hostile size; the budget must saturate rather than trap.
        Assert.True(ModelInstaller.RequiredFreeBytes(long.MaxValue) > 0);
        Assert.True(ModelInstaller.UnpackFreeBytes(long.MaxValue) > 0);
    }

    /// <summary>
    /// An archive entry that would unpack outside the destination is refused, and nothing it named
    /// is written.
    /// </summary>
    /// <remarks>
    /// The traversal guard the shelled-out <c>tar</c> gave the Swift target for free. Here it comes
    /// from <c>TarFile.ExtractToDirectory</c>, which is a framework promise rather than this
    /// library's code — which is exactly why it is worth a test. A framework behaviour nobody
    /// asserts is a framework behaviour nobody notices changing, and the thing being unpacked comes
    /// off the network.
    /// </remarks>
    [Theory]
    [InlineData("../escape.txt")]
    [InlineData("../../escape.txt")]
    [InlineData("nested/../../escape.txt")]
    public async Task AnArchiveEntryThatEscapesTheDestinationIsRefused(string entryName)
    {
        var entries = CompleteModel();
        entries[entryName] = "escaped";
        var archive = WriteArchive("evil.tar.gz", entries);
        var installer = InstallerFor(PinFor(archive, "the weights"));

        var failure = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.UnpackFailed, failure.Kind);
        Assert.Null(installer.InstalledVersion());
        Assert.Empty(Directory.EnumerateFiles(scratch, "escape.txt", SearchOption.AllDirectories));
    }
}

using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading;
using System.Threading.Tasks;

namespace PIIMasker;

/// <summary>Downloads, verifies, and installs the on-device model.</summary>
/// <remarks>
/// Fetches the <c>.tar.gz</c> a <see cref="ModelPin"/> names, checks its SHA-256, unpacks it,
/// checks the SHA-256 of the unpacked weights, and flips the promotion pointer atomically — so a
/// failed or tampered download can never replace a working install.
///
/// There is no manifest and no version negotiation: the pin is compiled in, so the only question
/// this asks is "are the bytes the pin names already on disk?".
/// </remarks>
public sealed class ModelInstaller
{
    /// <summary>The result of an install attempt.</summary>
    /// <param name="Directory">The directory holding the ready-to-load model.</param>
    /// <param name="Installed">
    /// False when the pinned version was already installed and complete: no download, no unpack, no
    /// swap — the bytes on disk are exactly the ones that were there before.
    /// </param>
    public sealed record Outcome(string Directory, bool Installed);

    /// <summary>Install progress, for a host driving a progress bar.</summary>
    public abstract record Progress
    {
        /// <summary>Deciding whether anything needs to be fetched at all.</summary>
        public sealed record Checking : Progress;

        /// <summary>Transferring the archive.</summary>
        /// <param name="Fraction">Completed share of the transfer, 0…1.</param>
        public sealed record Downloading(double Fraction) : Progress;

        /// <summary>Hashing the downloaded archive against the pin.</summary>
        public sealed record Verifying : Progress;

        /// <summary>Unpacking and promoting the verified archive.</summary>
        public sealed record Installing : Progress;

        /// <summary>Terminal: the pinned version was already on disk and nothing was written.</summary>
        /// <param name="Version">The version already installed.</param>
        public sealed record UpToDate(string Version) : Progress;

        /// <summary>Terminal: new bytes were written and promoted.</summary>
        /// <param name="Version">The version now installed.</param>
        public sealed record Installed(string Version) : Progress;
    }

    /// <summary>Why an install failed. The cases are distinct because each needs a different user action.</summary>
    public enum FailureKind
    {
        /// <summary>No usable network on this device. Distinct from a reachable network that cannot reach the host.</summary>
        Offline,
        /// <summary>The pin itself is unusable. A build defect rather than a runtime condition, and not fixable by retrying.</summary>
        InvalidPin,
        /// <summary>The bytes that arrived are not the bytes the pin names. A fresh download may fix it.</summary>
        ChecksumMismatch,
        /// <summary>The transfer failed for a reason that is neither offline nor a full disk.</summary>
        DownloadFailed,
        /// <summary>The archive arrived intact but could not be unpacked, or unpacked to the wrong contents.</summary>
        UnpackFailed,
        /// <summary>Not enough room on the volume for the archive and what it unpacks to.</summary>
        OutOfSpace,
        /// <summary>
        /// The payload was fine; this machine's model store could not be written. Distinct from
        /// <see cref="UnpackFailed"/> because re-downloading cannot fix it — the retry fails identically.
        /// </summary>
        StoreUnwritable,
    }

    /// <summary>An install failure, carrying the kind so a host can choose its copy and its action.</summary>
    /// <param name="kind">Which failure this is.</param>
    /// <param name="message">Log-facing detail. User-visible copy is the host's, chosen from <paramref name="kind"/>.</param>
    /// <param name="inner">The underlying failure, when there was one.</param>
    public sealed class InstallException(FailureKind kind, string message, Exception? inner = null)
        : Exception(message, inner)
    {
        /// <summary>Which failure this is, so a host can pick the copy and the action that fit.</summary>
        public FailureKind Kind { get; } = kind;
    }

    /// <summary>
    /// Upper bound on a published archive, checked when the pin is read. Two orders of magnitude
    /// above the real model, so it only ever catches a malformed pin.
    /// </summary>
    public const long MaxPlausibleArchiveBytes = 100_000_000_000;

    /// <summary>Everything a masker opens — keep this in step with the loader.</summary>
    /// <remarks>
    /// A directory holding only some of them must never count as installed: the loader fails, the
    /// fail-closed gate then holds every request, and nothing retries because both the
    /// installed-version check and the install itself thought they were done. Silent, unbadged,
    /// permanent.
    /// </remarks>
    public static IReadOnlyList<string> RequiredModelFiles { get; } = new ReadOnlyCollection<string>(
        ["model.onnx", "tokenizer.json", "tokenizer_config.json"]);

    /// <summary>Whether every required file is present in one directory.</summary>
    public static bool IsCompleteModelDir(string? directory) =>
        !string.IsNullOrEmpty(directory)
        && RequiredModelFiles.All(name => File.Exists(Path.Combine(directory, name)));

    private static readonly string[] PartialPrefixes = [".download-", ".staging-", ".current-", ".old-"];

    private readonly ModelPin pin;
    private readonly string installRoot;
    private readonly HttpClient http;

    /// <param name="installRoot">
    /// Where version directories and the promotion pointer are written. Required: a library has no
    /// business guessing a location inside the host's data directory.
    /// </param>
    /// <param name="pin">The model to install. Defaults to <see cref="ModelPin.Current"/>.</param>
    /// <param name="httpClient">Injectable for tests; otherwise a short-timeout client.</param>
    public ModelInstaller(string installRoot, ModelPin? pin = null, HttpClient? httpClient = null)
    {
        ArgumentException.ThrowIfNullOrEmpty(installRoot);
        this.installRoot = installRoot;
        this.pin = pin ?? ModelPin.Current;
        http = httpClient ?? DefaultClient.Value;
    }

    /// <summary>
    /// A short-timeout client so an unreachable host fails in seconds rather than hanging.
    /// </summary>
    /// <remarks>
    /// The 15-minute ceiling is a whole-transfer budget: without one a connection that stalls
    /// without dropping leaves the install spinning with no error and no way to retry.
    /// </remarks>
    private static readonly Lazy<HttpClient> DefaultClient = new(() => new HttpClient
    {
        Timeout = TimeSpan.FromMinutes(15),
    });

    /// <summary>
    /// True only for failures meaning "this device has no usable network" — not for a reachable
    /// network that cannot reach the model host.
    /// </summary>
    /// <remarks>
    /// <para>
    /// The distinction is what lets a host tell "no internet connection" from "we can't reach the
    /// download": the first is the user's network, the second is the publisher's. Getting it wrong
    /// sends the user to check a router that is working fine.
    /// </para>
    /// <para>
    /// <b>A DNS failure is NOT offline</b>, and this is the case that is easy to get wrong — it was,
    /// and a live install against an unresolvable host reported "No internet connection" on a machine
    /// with working internet. A name that will not resolve means a lapsed domain, a misconfigured
    /// CDN record, or a corporate resolver blocking us: every one of those is the publisher's problem
    /// wearing the user's clothes. The same goes for a host that is unreachable or refusing
    /// connections, and for a timeout. Only the adapter reporting that there is no network at all
    /// qualifies.
    /// </para>
    /// </remarks>
    public static bool IsOfflineError(Exception error) => Unwrap(error).Any(e =>
        e is SocketException s
        && s.SocketErrorCode is SocketError.NetworkDown or SocketError.NetworkUnreachable);

    /// <summary>A full disk, however it reaches us.</summary>
    /// <remarks>
    /// Windows reports <c>ERROR_DISK_FULL</c>/<c>ERROR_HANDLE_DISK_FULL</c> and Unix reports
    /// <c>ENOSPC</c>, either of which can sit on an inner exception rather than the top-level one —
    /// so walk the chain, as the Swift target walks the <c>NSError</c> chain.
    /// </remarks>
    public static bool IsOutOfSpaceError(Exception error) => Unwrap(error).Any(e =>
        e is IOException io && (io.HResult & 0xFFFF) is 0x27 or 0x70 or 28);

    /// <summary>A write refused for permissions rather than space.</summary>
    /// <remarks>
    /// This is the dominant read-only-store failure, and without recognising it that store reports
    /// as a network problem — sending the user to retry a download the store can never accept.
    /// </remarks>
    public static bool IsPermissionError(Exception error) => Unwrap(error).Any(e =>
        e is UnauthorizedAccessException
        || (e is IOException io && (io.HResult & 0xFFFF) is 5 or 13));

    private static IEnumerable<Exception> Unwrap(Exception error)
    {
        var seen = 0;
        for (var e = error; e != null && seen < 5; e = e.InnerException, seen++) yield return e;
    }

    /// <summary>The failure a transport error should surface as.</summary>
    /// <remarks>
    /// Collapsing every failure into "download failed" throws away the distinction — so a full disk
    /// arrives at the UI as "we couldn't reach the server" and the user retries a download that
    /// could never fit.
    /// </remarks>
    public static FailureKind ClassifyFailure(Exception error) =>
        IsOutOfSpaceError(error) ? FailureKind.OutOfSpace
        : IsPermissionError(error) ? FailureKind.StoreUnwritable
        : IsOfflineError(error) ? FailureKind.Offline
        : FailureKind.DownloadFailed;

    /// <summary>The directory holding the ready-to-load model, or null if nothing is installed yet.</summary>
    public string? CurrentModelDir => ModelStore.ResolvePromoted(installRoot);

    /// <summary>The installed version, read from the promotion pointer.</summary>
    public string? InstalledVersion()
    {
        var promoted = ModelStore.ResolvePromoted(installRoot);
        return promoted == null ? null : Path.GetFileName(promoted.TrimEnd(
            Path.DirectorySeparatorChar, Path.AltDirectorySeparatorChar));
    }

    /// <summary>
    /// Ensure the pinned model is installed; returns the model directory and whether this call is
    /// what put it there.
    /// </summary>
    /// <remarks>
    /// Callers need <see cref="Outcome.Installed"/> because "succeeded" and "wrote new bytes" are
    /// different answers: the up-to-date branch succeeds having changed nothing, so anything keyed
    /// on the model having changed — a stale load failure, a cache, a probe — must not fire for it.
    /// </remarks>
    /// <param name="force">
    /// Reinstall even when the installed version already matches — the recovery path for a model
    /// whose files are all present but which the loader cannot actually use. Without it the
    /// up-to-date branch reports success forever and the user has no way to replace it.
    /// </param>
    /// <param name="skipping">
    /// Versions a rollback refused. Only honoured while a usable model is already installed: with
    /// nothing on disk there is no working model to protect, and refusing the only pinned version
    /// would leave the host permanently without one.
    /// </param>
    /// <param name="onProgress">Called as the install advances, for a host driving a progress bar.</param>
    /// <param name="cancellationToken">Abandons the install; a partial download is swept by a later run.</param>
    public async Task<Outcome> EnsureLatestAsync(
        bool force = false,
        IReadOnlySet<string>? skipping = null,
        Action<Progress>? onProgress = null,
        CancellationToken cancellationToken = default)
    {
        var report = onProgress ?? (_ => { });
        report(new Progress.Checking());

        // Before the up-to-date return below: a kill mid-download leaves a ~140 MB `.download-`
        // file, and for a host whose model is already current that path is never reached again — so
        // the leak would be permanent, and it is exactly what turns the next model release into
        // "not enough storage".
        SweepPartials();

        if (!InstallSupport.IsSafePathComponent(pin.Version))
            throw new InstallException(FailureKind.InvalidPin, $"unsafe version: {pin.Version}");
        if (pin.Bytes <= 0 || pin.Bytes >= MaxPlausibleArchiveBytes)
            throw new InstallException(FailureKind.InvalidPin, $"implausible archive size: {pin.Bytes}");

        var installedDir = CurrentModelDir;
        if (skipping?.Contains(pin.Version) == true && IsCompleteModelDir(installedDir))
        {
            report(new Progress.UpToDate(InstalledVersion() ?? pin.Version));
            return new Outcome(installedDir!, Installed: false);
        }

        if (!force
            && string.Equals(InstalledVersion(), pin.Version, StringComparison.Ordinal)
            && IsCompleteModelDir(installedDir))
        {
            report(new Progress.UpToDate(pin.Version));
            return new Outcome(installedDir!, Installed: false);
        }

        try
        {
            Directory.CreateDirectory(installRoot);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new InstallException(FailureKind.StoreUnwritable, e.Message, e);
        }

        if (!HasRoomFor(pin.Bytes))
            throw new InstallException(FailureKind.OutOfSpace,
                $"needs ~{RequiredFreeBytes(pin.Bytes) / 1_000_000} MB, archive {pin.Bytes / 1_000_000} MB");

        // GUID-suffixed so two installs never write the same file — the versioned name alone would
        // let a second run clobber a first mid-download. `SweepPartials` clears any left behind.
        var archive = Path.Combine(installRoot, $".download-{pin.Version}-{Guid.NewGuid():N}.tar.gz");
        try
        {
            await DownloadAsync(pin.SourceUri, archive, report, cancellationToken).ConfigureAwait(false);

            report(new Progress.Verifying());
            var got = await InstallSupport.Sha256OfFileAsync(archive, cancellationToken).ConfigureAwait(false);
            if (!InstallSupport.DigestsMatch(pin.ArchiveSha256, got))
                throw new InstallException(FailureKind.ChecksumMismatch,
                    $"checksum mismatch (expected {pin.ArchiveSha256}, got {got})");

            report(new Progress.Installing());
            await InstallAsync(archive, cancellationToken).ConfigureAwait(false);
        }
        finally
        {
            TryDelete(archive);
        }

        report(new Progress.Installed(pin.Version));
        return new Outcome(Path.Combine(installRoot, pin.Version), Installed: true);
    }

    private async Task DownloadAsync(
        Uri source, string destination, Action<Progress> report, CancellationToken cancellationToken)
    {
        TryDelete(destination);
        if (source.IsFile)
        {
            File.Copy(source.LocalPath, destination, overwrite: true);
            report(new Progress.Downloading(1));
            return;
        }

        try
        {
            using var response = await http
                .GetAsync(source, HttpCompletionOption.ResponseHeadersRead, cancellationToken)
                .ConfigureAwait(false);
            response.EnsureSuccessStatusCode();

            var total = response.Content.Headers.ContentLength ?? pin.Bytes;
            await using var input = await response.Content.ReadAsStreamAsync(cancellationToken).ConfigureAwait(false);
            await using var output = new FileStream(
                destination, FileMode.Create, FileAccess.Write, FileShare.None,
                bufferSize: 1 << 20, useAsync: true);

            var buffer = new byte[1 << 20];
            long written = 0;
            var lastReported = -1;
            while (true)
            {
                var read = await input.ReadAsync(buffer, cancellationToken).ConfigureAwait(false);
                if (read == 0) break;
                await output.WriteAsync(buffer.AsMemory(0, read), cancellationToken).ConfigureAwait(false);
                written += read;
                // Report on whole percents only: a host repainting a progress bar per 1 MB chunk
                // would otherwise be handed ~140 callbacks a second on a fast link.
                var percent = total > 0 ? (int)(written * 100 / total) : 0;
                if (percent != lastReported)
                {
                    lastReported = percent;
                    report(new Progress.Downloading(total > 0 ? Math.Min(1, (double)written / total) : 0));
                }
            }
        }
        catch (Exception e) when (e is not OperationCanceledException and not InstallException)
        {
            TryDelete(destination);
            // A volume that filled mid-transfer can surface as a bare write failure with no
            // out-of-space code anywhere in the chain, which the classifier cannot tell from a
            // server problem. Here the volume is in scope, so ask it.
            if (LooksOutOfSpace(installRoot, pin.Bytes))
                throw new InstallException(FailureKind.OutOfSpace, "volume filled during the download", e);
            throw new InstallException(ClassifyFailure(e), e.Message, e);
        }
    }

    private async Task InstallAsync(string archive, CancellationToken cancellationToken)
    {
        var staging = Path.Combine(installRoot, $".staging-{pin.Version}-{Guid.NewGuid():N}");
        try
        {
            Directory.CreateDirectory(staging);
            await ExtractAsync(archive, staging, cancellationToken).ConfigureAwait(false);

            if (!IsCompleteModelDir(staging))
            {
                var present = Directory.Exists(staging)
                    ? string.Join(", ", Directory.GetFileSystemEntries(staging).Select(Path.GetFileName).Order())
                    : "nothing";
                throw new InstallException(FailureKind.UnpackFailed,
                    $"archive is missing one of {string.Join(", ", RequiredModelFiles)} — got: {present}");
            }

            // The weights, not just the archive. The archive hash is packer-dependent — bsdtar and
            // GNU tar produce different bytes from the same files — so it can only ever attest to
            // one publisher's tarball. The weights hash is the number the model card, the README
            // and the pin all quote, and it is what a reader can reproduce. Checked here, before
            // the swap, so a wrong-but-well-packed archive never becomes current.
            var weights = await InstallSupport
                .Sha256OfFileAsync(Path.Combine(staging, "model.onnx"), cancellationToken)
                .ConfigureAwait(false);
            if (!InstallSupport.DigestsMatch(pin.WeightsSha256, weights))
                throw new InstallException(FailureKind.ChecksumMismatch,
                    $"checksum mismatch (expected {pin.WeightsSha256}, got {weights})");

            PromoteStaging(staging);
        }
        finally
        {
            TryDeleteDirectory(staging);
        }
    }

    /// <summary>Unpack the <c>.tar.gz</c> in-process.</summary>
    /// <remarks>
    /// The Swift target shells out to <c>/usr/bin/tar</c>; .NET has <c>System.Formats.Tar</c> over
    /// <c>GZipStream</c> in the box, so there is no external process and no platform assumption
    /// about which tar is installed. <c>TarFile.ExtractToDirectory</c> refuses entries that would
    /// escape the destination, which is the traversal guard the shelled-out version got from bsdtar.
    /// </remarks>
    private async Task ExtractAsync(string archive, string destination, CancellationToken cancellationToken)
    {
        try
        {
            await using var file = new FileStream(
                archive, FileMode.Open, FileAccess.Read, FileShare.Read,
                bufferSize: 1 << 20, useAsync: true);
            await using var gzip = new GZipStream(file, CompressionMode.Decompress);
            await TarFile.ExtractToDirectoryAsync(
                gzip, destination, overwriteFiles: true, cancellationToken).ConfigureAwait(false);
        }
        catch (Exception e) when (e is not OperationCanceledException)
        {
            // tar failures name a full disk only indirectly, so ask the filesystem instead. What is
            // still needed at this point is the unpacked size — the archive is already on disk.
            if (LooksOutOfSpace(installRoot, UnpackFreeBytes(pin.Bytes)))
                throw new InstallException(FailureKind.OutOfSpace,
                    $"unpack needs ~{UnpackFreeBytes(pin.Bytes) / 1_000_000} MB", e);
            if (IsPermissionError(e))
                throw new InstallException(FailureKind.StoreUnwritable, e.Message, e);
            throw new InstallException(FailureKind.UnpackFailed, e.Message, e);
        }
    }

    /// <summary>Move a verified staging directory into its version directory, then promote it.</summary>
    /// <remarks>
    /// The version being replaced can be the one currently promoted — the repair path reinstalls the
    /// same version — so the old directory is moved aside rather than deleted first. A failure after
    /// a delete would leave the user with no model at all and a pointer aimed at nothing; moving
    /// aside means the worst case is recoverable and the old bytes are still there to restore.
    /// </remarks>
    private void PromoteStaging(string staging)
    {
        var versionDir = Path.Combine(installRoot, pin.Version);
        string? displaced = null;
        try
        {
            if (Directory.Exists(versionDir))
            {
                displaced = Path.Combine(installRoot, $".old-{pin.Version}-{Guid.NewGuid():N}");
                Directory.Move(versionDir, displaced);
            }
            Directory.Move(staging, versionDir);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            if (displaced != null && !Directory.Exists(versionDir))
            {
                try { Directory.Move(displaced, versionDir); displaced = null; }
                catch (IOException) { /* the restore failed too; the displaced copy is still on disk */ }
                catch (UnauthorizedAccessException) { /* as above */ }
            }
            throw new InstallException(FailureKind.StoreUnwritable, e.Message, e);
        }

        if (!IsCompleteModelDir(versionDir))
            throw new InstallException(FailureKind.StoreUnwritable, "installed dir is incomplete after the swap");

        PointCurrent(pin.Version, installRoot);
        if (displaced != null) TryDeleteDirectory(displaced);
    }

    /// <summary>Point the promotion pointer at a version directory, atomically.</summary>
    /// <remarks>
    /// <para>
    /// The Swift target writes a <c>current</c> symlink and renames it into place. This target
    /// writes a <c>current</c> plain-text file naming the version directory instead, because a
    /// symlink on Windows needs either elevation or Developer Mode — neither of which a per-user
    /// install can assume, and failing the promotion is how a verified model never becomes the one
    /// that loads. <see cref="ModelStore.ResolvePromoted"/> reads both layouts, so a store created
    /// by either target resolves.
    /// </para>
    /// <para>
    /// The write is temp-then-move: <see cref="File.Move(string, string, bool)"/> replaces within a
    /// volume atomically, so a reader sees either the old version or the new one and never a
    /// half-written pointer.
    /// </para>
    /// </remarks>
    public static void PointCurrent(string version, string installRoot)
    {
        if (!InstallSupport.IsSafePathComponent(version))
            throw new InstallException(FailureKind.StoreUnwritable, "unsafe version component");
        var current = Path.Combine(installRoot, "current");
        var temp = Path.Combine(installRoot, $".current-{Guid.NewGuid():N}");
        ClearDirectoryShapedCurrent(current);
        try
        {
            File.WriteAllText(temp, version);
            File.Move(temp, current, overwrite: true);
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            try { File.Delete(temp); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            throw new InstallException(FailureKind.StoreUnwritable, "could not update the promotion pointer", e);
        }
    }

    /// <summary>
    /// Make room for the pointer file when <c>current</c> is a directory entry rather than a file.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A store promoted by the Swift target has <c>current</c> as a SYMLINK, and
    /// <see cref="File.Move(string, string, bool)"/> will not replace a directory entry — so
    /// without this, a store either target can READ is a store only one of them can WRITE, and the
    /// install fails at the last step with a verified model already on disk.
    /// </para>
    /// <para>
    /// Only a symlink is removed, and removing a symlink deletes the link and never what it points
    /// at, so the version directory survives. A real directory named <c>current</c> is refused
    /// instead of deleted: <see cref="ModelStore.ResolvePromoted"/> accepts one as a model
    /// directory in its own right, so deleting it could destroy the very thing being promoted over.
    /// </para>
    /// </remarks>
    private static void ClearDirectoryShapedCurrent(string current)
    {
        if (!Directory.Exists(current)) return;
        var directory = new DirectoryInfo(current);
        if (directory.LinkTarget == null)
            throw new InstallException(FailureKind.StoreUnwritable,
                "`current` is a real directory rather than a promotion pointer — refusing to delete it");
        try
        {
            directory.Delete();
        }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException)
        {
            throw new InstallException(
                FailureKind.StoreUnwritable, "could not replace the `current` symlink", e);
        }
    }

    /// <summary>
    /// New disk the install needs at its peak: the downloaded archive (1×) plus what it unpacks to
    /// (~1.5× for this model), both present until the swap.
    /// </summary>
    /// <remarks>
    /// An already-installed version is NOT counted — it occupies space we already hold, so requiring
    /// headroom for it would refuse installs that fit.
    /// </remarks>
    public static long RequiredFreeBytes(long archiveBytes)
    {
        if (archiveBytes <= 0) return 0;
        // The pin may be caller-built; multiply-then-halve would overflow on a hostile value.
        // Saturate to large-but-sane rather than long.MaxValue, which would be quoted as
        // "9223372036854 MB" in a user-facing shortfall.
        try { return checked(archiveBytes * 5) / 2; }
        catch (OverflowException) { return long.MaxValue / 2; }
    }

    /// <summary>
    /// What the unpack alone needs, for quoting a shortfall discovered after the archive is already
    /// on disk.
    /// </summary>
    public static long UnpackFreeBytes(long archiveBytes)
    {
        if (archiveBytes <= 0) return 0;
        try { return checked(archiveBytes * 3) / 2; }
        catch (OverflowException) { return long.MaxValue / 2; }
    }

    /// <summary>
    /// Refuse before spending the bandwidth when the volume clearly cannot hold the result.
    /// Advisory only: an unreadable capacity is not treated as failure.
    /// </summary>
    private bool HasRoomFor(long bytes)
    {
        if (bytes <= 0) return true;
        var available = AvailableBytes(installRoot);
        return available == null || available >= RequiredFreeBytes(bytes);
    }

    /// <summary>Whether a write that already failed did so for want of space.</summary>
    /// <remarks>
    /// A volume with a few MB left is "full" for a 200 MB unpack, and telling that user their
    /// download was corrupted sends them retrying forever.
    /// </remarks>
    private static bool LooksOutOfSpace(string path, long needed)
    {
        var available = AvailableBytes(path);
        return available != null && available < needed;
    }

    /// <summary>Free bytes on the volume holding <paramref name="path"/>, or null if unreadable.</summary>
    /// <remarks>
    /// The Swift target additionally consults macOS's purgeable-space figure, which has no
    /// counterpart here and no meaning on Windows or Linux — so this is the single strict number.
    /// </remarks>
    private static long? AvailableBytes(string path)
    {
        try
        {
            var root = Path.GetPathRoot(Path.GetFullPath(path));
            return string.IsNullOrEmpty(root) ? null : new DriveInfo(root).AvailableFreeSpace;
        }
        catch (Exception e) when (e is ArgumentException or IOException or UnauthorizedAccessException)
        {
            return null;
        }
    }

    /// <summary>
    /// Remove partial downloads, staging directories and displaced versions an interrupted run left
    /// behind. Installed version directories and the promotion pointer never start with a dot, so
    /// they are untouched.
    /// </summary>
    /// <remarks>
    /// Age-gated, because an install root can be shared: a second process installing at this moment
    /// has its own in-flight entries here, and reaping those breaks an install that is going fine.
    /// Anything genuinely abandoned is minutes old; anything live is seconds.
    /// </remarks>
    private void SweepPartials()
    {
        if (!Directory.Exists(installRoot)) return;
        var cutoff = DateTime.UtcNow.AddMinutes(-15);
        IEnumerable<string> entries;
        try { entries = Directory.EnumerateFileSystemEntries(installRoot); }
        catch (Exception e) when (e is IOException or UnauthorizedAccessException) { return; }

        foreach (var entry in entries)
        {
            var name = Path.GetFileName(entry);
            if (!PartialPrefixes.Any(p => name.StartsWith(p, StringComparison.Ordinal))) continue;
            try
            {
                var modified = Directory.Exists(entry)
                    ? Directory.GetLastWriteTimeUtc(entry)
                    : File.GetLastWriteTimeUtc(entry);
                if (modified >= cutoff) continue;
            }
            catch (Exception e) when (e is IOException or UnauthorizedAccessException) { continue; }

            if (Directory.Exists(entry)) TryDeleteDirectory(entry); else TryDelete(entry);
        }
    }

    private static void TryDelete(string path)
    {
        try { if (File.Exists(path)) File.Delete(path); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private static void TryDeleteDirectory(string path)
    {
        try { if (Directory.Exists(path)) Directory.Delete(path, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }
}

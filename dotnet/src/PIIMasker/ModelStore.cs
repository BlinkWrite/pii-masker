using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.IO;
using System.Threading;

namespace PIIMasker;

/// <summary>Where the model lives, and what the loader last concluded about it.</summary>
/// <remarks>
/// Everything here is process-wide on purpose. The installer writes bytes; a long-lived masker
/// holds an inference session built from the bytes that were there when it loaded; a host's status
/// badge watches for changes. Those three never meet, and per-instance state would let them
/// disagree — most dangerously after a rollback, where a masker would keep masking with the very
/// version the probe just rejected. See <see cref="Generation"/>.
/// </remarks>
public static class ModelStore
{
    /// <summary>Where a model is looked for, in order.</summary>
    /// <remarks>
    /// The first directory holding ALL of <see cref="ModelInstaller.RequiredModelFiles"/> wins.
    /// Answering per file instead would let a directory missing one of them still report installed
    /// by resolving that file from a different candidate — and the loader, which reads them from a
    /// single folder, would then fail with nothing to retry it.
    /// </remarks>
    /// <param name="InstallRoot">The root <see cref="ModelInstaller"/> writes into, or null.</param>
    /// <param name="Fallbacks">Searched after the install root, in order — a copy shipped with the host, a development checkout.</param>
    public sealed record Location(string? InstallRoot, IReadOnlyList<string> Fallbacks)
    {
        /// <summary>One install root and no fallbacks.</summary>
        /// <param name="installRoot">The root the installer writes into, or null for unconfigured.</param>
        public Location(string? installRoot) : this(installRoot, Array.Empty<string>()) { }

        /// <summary>
        /// Nothing configured. Every lookup answers "no model", so a caller's fail-closed gate holds
        /// everything — the correct behaviour for a host that forgot to call <see cref="Configure(Location)"/>.
        /// </summary>
        // The cast disambiguates the string overload from the record's own copy constructor.
        public static Location Unconfigured { get; } = new((string?)null);
    }

    private static Location location = Location.Unconfigured;
    private static bool installedCache;
    private static int generation;
    private static State state = State.Unknown;

    /// <summary>The configured search location. Set once at startup, before any masker is built.</summary>
    public static Location CurrentLocation => location;

    /// <summary>
    /// Point the library at a model store. Call once, at startup. Re-configuring invalidates
    /// everything the loader concluded about the previous location.
    /// </summary>
    public static void Configure(Location newLocation)
    {
        ArgumentNullException.ThrowIfNull(newLocation);
        location = newLocation;
        Invalidate();
    }

    /// <summary>Convenience for the common shape: one install root, optional fallbacks.</summary>
    public static void Configure(string installRoot, IReadOnlyList<string>? fallbacks = null) =>
        Configure(new Location(installRoot, fallbacks ?? Array.Empty<string>()));

    /// <summary>The one directory every model file is loaded from, or null if no candidate is complete.</summary>
    public static string? ResolvedModelDirectory()
    {
        var snapshot = location;
        if (snapshot.InstallRoot is { Length: > 0 } root)
        {
            var promoted = ResolvePromoted(root);
            if (promoted != null && ModelInstaller.IsCompleteModelDir(promoted)) return promoted;
        }
        foreach (var fallback in snapshot.Fallbacks)
            if (ModelInstaller.IsCompleteModelDir(fallback)) return fallback;
        return null;
    }

    /// <summary>
    /// The version directory currently promoted under <paramref name="installRoot"/>.
    /// </summary>
    /// <remarks>
    /// Two layouts are accepted, because one of them cannot exist everywhere. The Swift target
    /// promotes by pointing a <c>current</c> symlink at a version directory, and .NET follows a
    /// symlink transparently, so that layout resolves here unchanged. Windows cannot create a
    /// symlink without either elevation or Developer Mode — neither of which a per-user install can
    /// assume — so a <c>current</c> plain-text file naming the version directory is accepted as
    /// well. Both are atomic to replace; the file form is a rename within the volume.
    ///
    /// The named version is checked with <see cref="InstallSupport.IsSafePathComponent"/> before it
    /// is joined, because the file is on disk and a caller that can write it must not be able to
    /// aim the loader — or a later recursive delete — outside the install root.
    /// </remarks>
    public static string? ResolvePromoted(string installRoot)
    {
        ArgumentException.ThrowIfNullOrEmpty(installRoot);
        var current = Path.Combine(installRoot, "current");
        if (Directory.Exists(current)) return current;
        if (!File.Exists(current)) return null;
        string named;
        try
        {
            named = File.ReadAllText(current).Trim();
        }
        catch (IOException) { return null; }
        catch (UnauthorizedAccessException) { return null; }
        return InstallSupport.IsSafePathComponent(named) ? Path.Combine(installRoot, named) : null;
    }

    /// <summary>Whether a model is present.</summary>
    /// <remarks>
    /// A cheap disk check a caller's per-request gate uses to avoid sending unmasked context before
    /// the model has been installed. Once the model is on disk it stays there for the process's
    /// life, so the positive result is cached: this runs many times a second while a user types and
    /// otherwise stats three files each call. Only false→true is cached — a monotonic transition —
    /// so a benign race just repeats the disk check.
    /// </remarks>
    public static bool ModelIsInstalled()
    {
        if (Volatile.Read(ref installedCache)) return true;
        var installed = ResolvedModelDirectory() != null;
        if (installed) Volatile.Write(ref installedCache, true);
        return installed;
    }

    /// <summary>What the loader knows about the model on disk.</summary>
    /// <remarks>
    /// Only the loader may report <see cref="Loaded"/> or <see cref="Unusable"/>, because only the
    /// loader has actually opened the files — everyone else can invalidate but never certify. That
    /// asymmetry is the point: letting the installer clear the failure flag directly means a refresh
    /// that found the model already up to date and wrote nothing still declares it healthy, putting
    /// an unloadable model back into silence. <see cref="Unknown"/> is "no load attempted since the
    /// bytes last changed", not "fine".
    /// </remarks>
    public enum State
    {
        /// <summary>No load attempted since the bytes last changed. Not the same as "fine".</summary>
        Unknown,
        /// <summary>The loader opened every model file successfully.</summary>
        Loaded,
        /// <summary>Every file was present, but the loader could not open them.</summary>
        Unusable,
    }

    /// <summary>The loader's last conclusion. Raises <see cref="StateChanged"/> on a real change.</summary>
    public static State CurrentState
    {
        get => state;
        private set
        {
            if (state == value) return;
            state = value;
            StateChanged?.Invoke(null, EventArgs.Empty);
        }
    }

    /// <summary>
    /// Raised when the loader's conclusion moves, so a host's status badge is recomputed at once.
    /// </summary>
    /// <remarks>
    /// The load runs asynchronously off launch, so whether it lands before or after the host's last
    /// refresh is a race — and it is lost whenever the inference session builds successfully and
    /// only the tokenizer fails. Without this push the user sees a healthy-looking app with no badge
    /// and no retry, silently holding every request.
    /// </remarks>
    public static event EventHandler? StateChanged;

    /// <summary>
    /// Bumped whenever the bytes on disk change. Maskers compare it against the generation they
    /// loaded at, so a session built from the old bytes is dropped rather than reused.
    /// </summary>
    public static int Generation => Volatile.Read(ref generation);

    /// <summary>
    /// The bytes on disk changed, so whatever the loader last concluded no longer describes them —
    /// and neither does any session already built from them.
    /// </summary>
    public static void Invalidate()
    {
        CurrentState = State.Unknown;
        Volatile.Write(ref installedCache, false);
        Interlocked.Increment(ref generation);
    }

    /// <summary>
    /// Reported by the loader when the files were all present but could not be opened.
    /// </summary>
    /// <remarks>
    /// Nothing else can detect this — <see cref="ModelIsInstalled"/> only sees the files — so
    /// without recording it the host looks healthy while masking nothing, with no affordance to
    /// recover.
    /// </remarks>
    public static void MarkUnusable()
    {
        CurrentState = State.Unusable;
        Volatile.Write(ref installedCache, false);
    }

    /// <summary>Reported by the loader when the model opened successfully.</summary>
    public static void MarkLoaded() => CurrentState = State.Loaded;
}

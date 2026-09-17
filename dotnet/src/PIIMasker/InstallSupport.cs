using System;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Threading;
using System.Threading.Tasks;

namespace PIIMasker;

/// <summary>
/// Two filesystem helpers the installer needs: a streaming SHA-256 of a file, and a
/// path-component safety check for a version string before it is used as a directory name.
/// </summary>
public static class InstallSupport
{
    /// <summary>
    /// Whether a string is safe to use verbatim as one on-disk path component.
    /// </summary>
    /// <remarks>
    /// <para>
    /// A version becomes a path component (the install version directory, the promotion target),
    /// so a slash or <c>..</c> would escape the install root — turning a recursive delete of the
    /// version directory into a recursive delete of an arbitrary one. A <see cref="ModelPin"/> is
    /// public and a caller can build one, so this is checked at every filesystem use rather than
    /// assumed.
    /// </para>
    /// <para>
    /// This target additionally rejects the Windows reserved device names, which the Swift target
    /// has no reason to consider. <c>CON</c>, <c>NUL</c>, <c>COM1</c> and the rest are not
    /// filenames on Windows at any extension: creating <c>NUL.2026</c> silently resolves to the
    /// device rather than a directory, so an installer would write a model into nothing and then
    /// fail to read it back with no indication why.
    /// </para>
    /// </remarks>
    public static bool IsSafePathComponent(string? value)
    {
        if (string.IsNullOrWhiteSpace(value) || value is "." or "..") return false;
        if (!value.All(c => c is >= 'A' and <= 'Z' or >= 'a' and <= 'z' or >= '0' and <= '9'
                or '.' or '_' or '-'))
            return false;
        var stem = value.Split('.')[0];
        return !WindowsReservedNames.Contains(stem, StringComparer.OrdinalIgnoreCase);
    }

    private static readonly string[] WindowsReservedNames =
    [
        "CON", "PRN", "AUX", "NUL",
        "COM1", "COM2", "COM3", "COM4", "COM5", "COM6", "COM7", "COM8", "COM9",
        "LPT1", "LPT2", "LPT3", "LPT4", "LPT5", "LPT6", "LPT7", "LPT8", "LPT9",
    ];

    /// <summary>Lowercase hex SHA-256 of a file, streamed rather than read whole.</summary>
    /// <remarks>
    /// The archive is ~137 MB and the weights ~188 MiB, so neither is loaded into memory. The
    /// buffer matches the Swift target's 1 MB chunk.
    /// </remarks>
    public static async Task<string> Sha256OfFileAsync(string path, CancellationToken cancellationToken = default)
    {
        await using var stream = new FileStream(
            path, FileMode.Open, FileAccess.Read, FileShare.Read,
            bufferSize: 1 << 20, useAsync: true);
        using var sha = SHA256.Create();
        var digest = await sha.ComputeHashAsync(stream, cancellationToken).ConfigureAwait(false);
        return Convert.ToHexString(digest).ToLowerInvariant();
    }

    /// <summary>
    /// Whether two hex digests are the same, without leaking where they first differ through timing.
    /// </summary>
    /// <remarks>
    /// An ordinary string comparison returns as soon as two characters differ, which over enough
    /// attempts tells an attacker who controls the served bytes how much of a prefix they have
    /// matched. That is a weak channel here — the comparison runs once per download, not per
    /// guess — but a fixed-time check costs nothing and removes the question.
    /// </remarks>
    public static bool DigestsMatch(string? expected, string? actual)
    {
        if (expected is null || actual is null || expected.Length != actual.Length) return false;
        var difference = 0;
        for (var i = 0; i < expected.Length; i++)
            difference |= char.ToLowerInvariant(expected[i]) ^ char.ToLowerInvariant(actual[i]);
        return difference == 0;
    }
}

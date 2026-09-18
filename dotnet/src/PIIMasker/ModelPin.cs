using System;
using System.Collections.Generic;
using System.Collections.ObjectModel;
using System.Linq;

namespace PIIMasker;

/// <summary>
/// SHA-256 identities for the loose files a loader opens directly, alongside the weights.
/// </summary>
/// <remarks>
/// Absent from the Swift target, which reaches the tokenizer through swift-transformers and never
/// opens the files itself. A .NET host reads <c>tokenizer.json</c> and <c>tokenizer_config.json</c>
/// off disk, so their bytes are part of the accepted identity here rather than an implementation
/// detail. Carried on the pin, not in <see cref="MaskerConfig"/>: these say which model this is,
/// not how it should behave.
/// </remarks>
/// <param name="TokenizerSha256">SHA-256 of <c>tokenizer.json</c>.</param>
/// <param name="TokenizerConfigSha256">SHA-256 of <c>tokenizer_config.json</c>.</param>
public sealed record ModelFileManifest(string TokenizerSha256, string TokenizerConfigSha256);

/// <summary>One published model release, pinned in source.</summary>
/// <remarks>
/// <para>
/// The version is not fetched at runtime. A hosted manifest's SHA-256 makes the <i>weights</i> host
/// untrusted, but whoever serves the manifest sets that hash — so a manifest relocates trust rather
/// than removing it. A value compiled into the library has no host at all. The cost is that a new
/// model needs a new library version and a host-app release, which for a library is the normal
/// contract.
/// </para>
/// <para>
/// <see cref="SourceUri"/> must be immutable. For Hugging Face that means
/// <c>resolve/&lt;commit-sha&gt;/…</c>, never <c>resolve/main</c> — an entry in <see cref="Known"/>
/// has to stay fetchable after a newer model lands, which is what makes rollback to an older entry
/// work.
/// </para>
/// </remarks>
public sealed record ModelPin
{
    /// <summary>
    /// Used verbatim as the install directory name and the promotion target, so it must pass
    /// <see cref="InstallSupport.IsSafePathComponent"/>.
    /// </summary>
    public required string Version { get; init; }

    /// <summary>Where the archive is fetched from. Must be immutable — see the type remarks.</summary>
    public required Uri SourceUri { get; init; }

    /// <summary>SHA-256 of the <c>.tar.gz</c> at <see cref="SourceUri"/>, checked before it is unpacked.</summary>
    public required string ArchiveSha256 { get; init; }

    /// <summary>
    /// SHA-256 of the unpacked <c>model.onnx</c>, checked after the unpack. This is the number that
    /// goes in the README and the model card: the archive hash is packer-dependent (bsdtar and GNU
    /// tar disagree), while the weights hash is the same everywhere.
    /// </summary>
    public required string WeightsSha256 { get; init; }

    /// <summary>Archive size. Drives the disk-space budget and the figure quoted to the user.</summary>
    public required long Bytes { get; init; }

    /// <summary>
    /// Must equal this model's <c>config.max_width</c>. It lives on the pin rather than in
    /// <see cref="MaskerConfig"/> because it is a property of the weights, not a taste knob — which
    /// is exactly what makes reverting to an older entry safe.
    /// </summary>
    public required int MaxWidth { get; init; }

    /// <summary>
    /// Must equal this model's <c>config.max_len</c> — the longest input sequence it was trained to
    /// handle, counted in tokens (the label preamble included, since that shares the window).
    /// </summary>
    /// <remarks>
    /// A SAFETY limit, not a performance one. GLiNER uses relative position embeddings, so an
    /// over-length input does not error — it degrades. Measured on the 2026.08.1 weights, recall
    /// falls off past this many tokens and reaches <i>zero</i> around 1,250: the model reports no
    /// entities, the sanitizer reads that as "nothing to mask", and the caller sends raw PII
    /// believing it was masked. That is the one way this library can fail open, so the length is
    /// checked against this number and the pass is dropped instead. Too low only costs dropped
    /// requests; too high reopens the hole — so there is no default.
    /// </remarks>
    public required int MaxSequenceLength { get; init; }

    /// <summary>Identities for the loose files this target opens directly.</summary>
    public required ModelFileManifest Files { get; init; }

    /// <summary>The same pin, fetched from somewhere else — a mirror, a corporate proxy, a <c>file://</c> path.</summary>
    /// <remarks>
    /// Both hashes are carried over deliberately, and that is the whole point: they are hashes of
    /// the <i>bytes</i>, not of the address, so the same archive served from anywhere still has to
    /// match what this library was built against. The URI decides where to look; the hashes decide
    /// whether to trust what comes back. Overriding the hashes too would mean trusting your own
    /// host, which is the thing pinning exists to avoid, so that is left to constructing a pin
    /// outright — where it reads as the deliberate act it is.
    /// </remarks>
    public ModelPin WithSourceUri(Uri sourceUri)
    {
        ArgumentNullException.ThrowIfNull(sourceUri);
        if (!sourceUri.IsAbsoluteUri)
            throw new ArgumentException("A model source URI must be absolute.", nameof(sourceUri));
        return this with { SourceUri = sourceUri };
    }

    /// <summary>
    /// GLiNER small PII, ONNX INT8. Uploaded 21 Aug 2026; every field verified against the live
    /// repository at this revision, and mirrored in <c>model.json</c> at the root of this
    /// repository so a reader can diff the two without building anything.
    /// </summary>
    /// <remarks>
    /// The underscores are deliberate and the analyzer is overruled rather than obeyed: this name
    /// is the model version <c>2026.08.1</c>, spelled the only way a C# identifier can spell it.
    /// The Swift target calls the same release <c>v2026_08_1</c>; renaming it here to satisfy a
    /// naming rule would make the two targets disagree about what a release is called, which is
    /// precisely what a shared repository exists to prevent.
    /// </remarks>
    [System.Diagnostics.CodeAnalysis.SuppressMessage(
        "Naming", "CA1707:Identifiers should not contain underscores",
        Justification = "The identifier is a model version, matching the Swift target's spelling.")]
    public static ModelPin V2026_08_1 { get; } = new()
    {
        Version = "2026.08.1",
        SourceUri = new Uri(
            "https://huggingface.co/blinkwrite-ai/gliner-small-pii-onnx-int8/resolve/"
            + "0f079cbb338b2de74cda307fc10c93409daa1747/gliner-pii-2026.08.1.tar.gz"),
        ArchiveSha256 = "6dab57bc8f550b4e18b16b2b6d5a5de60712c5b5d34686f43873617e270b0a26",
        WeightsSha256 = "2ac41b218b8a87aaf06222fe6431e04b7b2cccb1acc41d1009696ce455014ef7",
        Bytes = 143_492_231,
        MaxWidth = 12,
        MaxSequenceLength = 768,
        Files = new ModelFileManifest(
            TokenizerSha256: "07d65756a669ff551b99568f8cf6e34a2937462816a15dd77d1c43ee78d2372c",
            TokenizerConfigSha256: "a3c213a21a411d13a28b10f037219d95ba24ab8ebf2469ec647edabb3bf75097"),
    };

    /// <summary>
    /// Every model this library knows how to install, oldest first. Rollback moves between entries,
    /// so nothing is ever removed from this list — only appended.
    /// </summary>
    public static IReadOnlyList<ModelPin> Known { get; } = new ReadOnlyCollection<ModelPin>([V2026_08_1]);

    /// <summary>What a fresh install gets. <see cref="Known"/> is never empty, so this is total.</summary>
    public static ModelPin Current => Known[^1];

    /// <summary>
    /// The entry immediately before <paramref name="version"/> in <see cref="Known"/>, or null if it
    /// is the oldest (or unknown). This is what a rollback reverts <i>to</i> when the previous
    /// version's bytes are not already on disk.
    /// </summary>
    public static ModelPin? Predecessor(string version)
    {
        var index = -1;
        for (var i = 0; i < Known.Count; i++)
            if (string.Equals(Known[i].Version, version, StringComparison.Ordinal)) { index = i; break; }
        return index > 0 ? Known[index - 1] : null;
    }
}

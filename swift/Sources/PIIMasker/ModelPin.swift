import Foundation

/// One published model release, pinned in source.
///
/// The version is not fetched at runtime. A hosted manifest's SHA-256 makes the *weights* host
/// untrusted, but whoever serves the manifest sets that hash — so a manifest relocates trust
/// rather than removing it. A value compiled into the library has no host at all. The cost is
/// that a new model needs a new library version and a host-app release, which for a library is
/// the normal contract.
///
/// `sourceURL` must be immutable. For Hugging Face that means `resolve/<commit-sha>/…`, never
/// `resolve/main` — an entry in ``known`` has to stay fetchable after a newer model lands, which
/// is what makes rollback to an older entry work.
public struct ModelPin: Sendable, Equatable {
    /// Used verbatim as the install directory name and the `current` symlink target, so it must
    /// pass ``InstallSupport/isSafePathComponent(_:)``.
    public let version: String
    public let sourceURL: URL
    /// SHA-256 of the `.tar.gz` at `sourceURL`, checked before it is unpacked.
    public let archiveSHA256: String
    /// SHA-256 of the unpacked `model.onnx`, checked after the unpack. This is the number that
    /// goes in the README and the model card: the archive hash is packer-dependent (bsdtar and
    /// GNU tar disagree), while the weights hash is the same everywhere.
    public let weightsSHA256: String
    /// Archive size. Drives the disk-space budget and the figure quoted to the user.
    public let bytes: Int64
    /// Must equal this model's `config.max_width`. It lives on the pin rather than in
    /// ``MaskerConfig`` because it is a property of the weights, not a taste knob — which is
    /// exactly what makes reverting to an older entry safe.
    public let maxWidth: Int
    /// Must equal this model's `config.max_len` — the longest input sequence it was trained to
    /// handle, counted in tokens (the label preamble included, since that shares the window).
    ///
    /// This is a SAFETY limit, not a performance one. GLiNER uses relative position embeddings, so
    /// an over-length input does not error — it degrades. Measured on the 2026.08.1 weights, recall
    /// falls off past this many tokens and reaches *zero* around 1,250: the model reports no
    /// entities, ``PrivacyFilter/sanitize(_:protecting:userNameMasking:)`` reads that as "nothing to
    /// mask", and the caller sends raw PII believing it was masked. That is the one way this library
    /// can fail open, so the length is checked against this number and the pass is dropped instead.
    ///
    /// On the pin for the same reason as ``maxWidth``: it is a property of the weights. Too low only
    /// costs dropped requests; too high reopens the hole — so there is no default.
    public let maxSequenceLength: Int

    public init(
        version: String, sourceURL: URL, archiveSHA256: String, weightsSHA256: String,
        bytes: Int64, maxWidth: Int, maxSequenceLength: Int
    ) {
        self.version = version
        self.sourceURL = sourceURL
        self.archiveSHA256 = archiveSHA256
        self.weightsSHA256 = weightsSHA256
        self.bytes = bytes
        self.maxWidth = maxWidth
        self.maxSequenceLength = maxSequenceLength
    }

    /// The same pin, fetched from somewhere else — a mirror, a corporate proxy, a `file://` path.
    ///
    /// Both hashes are carried over deliberately, and that is the whole point: they are hashes of
    /// the *bytes*, not of the address, so the same archive served from anywhere still has to match
    /// what this library was built against. The URL decides where to look; the hashes decide
    /// whether to trust what comes back.
    ///
    /// Overriding the hashes too means trusting your own host, which is the thing pinning exists to
    /// avoid — so that is left to the full ``init(version:sourceURL:archiveSHA256:weightsSHA256:bytes:maxWidth:)``,
    /// where it reads as the deliberate act it is.
    public func withSourceURL(_ url: URL) -> ModelPin {
        ModelPin(
            version: version, sourceURL: url, archiveSHA256: archiveSHA256,
            weightsSHA256: weightsSHA256, bytes: bytes, maxWidth: maxWidth,
            maxSequenceLength: maxSequenceLength)
    }
}

extension ModelPin {
    /// GLiNER small PII, ONNX INT8. Uploaded 21 Aug 2026; every field verified against the live
    /// repository at this revision, and mirrored in `model.json` at the root of this package so a
    /// reader can diff the two without building anything.
    public static let v2026_08_1 = ModelPin(
        version: "2026.08.1",
        sourceURL: URL(
            string: "https://huggingface.co/blinkwrite-ai/gliner-small-pii-onnx-int8"
                + "/resolve/0f079cbb338b2de74cda307fc10c93409daa1747"
                + "/gliner-pii-2026.08.1.tar.gz")!,
        archiveSHA256: "6dab57bc8f550b4e18b16b2b6d5a5de60712c5b5d34686f43873617e270b0a26",
        weightsSHA256: "2ac41b218b8a87aaf06222fe6431e04b7b2cccb1acc41d1009696ce455014ef7",
        bytes: 143_492_231,
        maxWidth: 12,
        maxSequenceLength: 768
    )

    /// Every model this library knows how to install, oldest first. Rollback moves between
    /// entries, so nothing is ever removed from this list — only appended.
    public static let known: [ModelPin] = [.v2026_08_1]

    /// What a fresh install gets. `known` is never empty, so this is total.
    public static var current: ModelPin { known[known.count - 1] }

    /// The entry immediately before `version` in ``known``, or nil if it is the oldest (or
    /// unknown). This is what a rollback reverts *to* when the previous version's bytes are not
    /// already on disk.
    public static func predecessor(of version: String) -> ModelPin? {
        guard let i = known.firstIndex(where: { $0.version == version }), i > 0 else { return nil }
        return known[i - 1]
    }
}

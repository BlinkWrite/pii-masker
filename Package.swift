// swift-tools-version:5.10
import PackageDescription

let package = Package(
    name: "swift-pii-masker",
    // macOS only, deliberately. The sources are portable (Foundation, CryptoKit, os, and the
    // three ML modules — no AppKit), and swift-transformers would allow iOS 16+. What is missing
    // for iOS is not API but evidence: a second CI runner, an install root that isn't
    // Application Support, and ONNX Runtime memory budgets nobody has measured on a phone.
    // Widening `platforms:` later is additive and breaks no caller.
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "PIIMasker", targets: ["PIIMasker"]),
        // Pipe text through the masker and see what it catches, without writing any Swift.
        .executable(name: "pii-mask", targets: ["pii-mask"]),
    ],
    dependencies: [
        // Brings both products this package imports: `Tokenizers` for the GLiNER tokenizer and
        // `Hub` for reading `tokenizer.json` / `tokenizer_config.json` off disk. `Hub` is declared
        // explicitly below rather than left to ride in behind `Tokenizers` as a transitive import.
        // swift-huggingface arrives through this package, so it is not declared here.
        .package(url: "https://github.com/huggingface/swift-transformers", from: "1.3.2"),
        .package(url: "https://github.com/microsoft/onnxruntime-swift-package-manager", from: "1.22.0"),
    ],
    // Every target carries an explicit `path` because the sources now live under `swift/`, one
    // folder per language target — see the root README.
    //
    // This manifest itself stays at the repository root and cannot move: SwiftPM resolves a git
    // dependency by reading `Package.swift` there, with no way to point a repository URL at a
    // subdirectory. Moving it would break every consumer pinning this repository, BlinkWrite's
    // macOS app included.
    targets: [
        .target(
            name: "PIIMasker",
            dependencies: [
                .product(name: "Hub", package: "swift-transformers"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "onnxruntime", package: "onnxruntime-swift-package-manager"),
            ],
            path: "swift/Sources/PIIMasker"
        ),
        .executableTarget(
            name: "pii-mask",
            dependencies: ["PIIMasker"],
            path: "swift/Sources/pii-mask"
        ),
        .testTarget(
            name: "PIIMaskerTests",
            dependencies: ["PIIMasker"],
            path: "swift/Tests/PIIMaskerTests"
        ),
    ]
)

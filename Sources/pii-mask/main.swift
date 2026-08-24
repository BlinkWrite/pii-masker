import Foundation
import PIIMasker

// A small pipe-text-through-it tool, so anyone can see what the masker catches without writing a
// line of Swift. Reads stdin, prints the masked text, and — with --show-map — what each placeholder
// stood for.
//
//   echo "email me at a@b.com" | pii-mask --model /path/to/gliner
//
// Deliberately prints the RESTORE MAP only on request: the whole point of the tool is to show what
// would leave the machine, and dumping the originals next to it undermines reading that at a glance.

struct Options {
    var modelDir: URL?
    var showMap = false
    var timeout: TimeInterval = 30
    var threshold: Float?
}

/// `--help` is a successful request for the help text, so it goes to stdout and exits 0; a usage
/// error goes to stderr and exits 1. Conflating the two makes `pii-mask --help | less` print
/// nothing and breaks any script that checks the status.
func usage(asked: Bool = false) -> Never {
    let text = """
        pii-mask — pipe text through the masker and see what it catches.

        USAGE
          pii-mask [--model DIR] [--show-map] [--threshold N] [--timeout SECONDS]

          Reads text on stdin and writes the masked text to stdout.

        OPTIONS
          --model DIR      Directory holding model.onnx, tokenizer.json and
                           tokenizer_config.json. Defaults to $PII_MASKER_MODEL_DIR.
          --show-map       Also print each placeholder and the text it replaced, on stderr.
          --threshold N    Score floor for keeping a span (default \(MaskerConfig.default.threshold)).
                           Lower catches more and false-positives more.
          --timeout N      Seconds the pass may take (default \(Options().timeout)).

        EXIT STATUS
          0  masked
          1  bad usage, or no model
          2  the masker produced nothing — a caller would send nothing here.
             Includes input over the token budget. Text longer than the model's
             window is split into several passes, not dropped.

        """
    let handle = asked ? FileHandle.standardOutput : FileHandle.standardError
    handle.write(text.data(using: .utf8)!)
    exit(asked ? 0 : 1)
}

var opts = Options()
var args = Array(CommandLine.arguments.dropFirst())
while let arg = args.first {
    args.removeFirst()
    switch arg {
    case "--model":
        guard let v = args.first else { usage() }
        args.removeFirst()
        opts.modelDir = URL(fileURLWithPath: v, isDirectory: true)
    case "--show-map":
        opts.showMap = true
    case "--threshold":
        guard let v = args.first, let n = Float(v) else { usage() }
        args.removeFirst()
        opts.threshold = n
    case "--timeout":
        guard let v = args.first, let n = TimeInterval(v) else { usage() }
        args.removeFirst()
        opts.timeout = n
    case "-h", "--help":
        usage(asked: true)
    default:
        FileHandle.standardError.write("unknown argument: \(arg)\n".data(using: .utf8)!)
        usage()
    }
}

let envDir = ProcessInfo.processInfo.environment["PII_MASKER_MODEL_DIR"]
    .flatMap { $0.isEmpty ? nil : URL(fileURLWithPath: $0, isDirectory: true) }
guard let modelDir = opts.modelDir ?? envDir else {
    FileHandle.standardError.write(
        "no model: pass --model DIR or set PII_MASKER_MODEL_DIR\n".data(using: .utf8)!)
    exit(1)
}
guard ModelInstaller.isCompleteModelDir(modelDir) else {
    FileHandle.standardError.write("""
        \(modelDir.path) is missing one of \(ModelInstaller.requiredModelFiles.joined(separator: ", ")).
        Unpack the archive ModelPin.current names, or fetch the loose files from the same revision.

        """.data(using: .utf8)!)
    exit(1)
}

let input = String(data: FileHandle.standardInput.readDataToEndOfFile(), encoding: .utf8) ?? ""
guard !input.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { exit(0) }

let config = MaskerConfig(threshold: opts.threshold ?? MaskerConfig.default.threshold)
let masker = PrivacyFilter(config: config, modelDirectory: modelDir)

// The masker is an actor and this is a plain executable, so hop into an async context and wait.
let done = DispatchSemaphore(value: 0)
nonisolated(unsafe) var status: Int32 = 0
Task {
    defer { done.signal() }
    guard let out = await masker.maskFields([input], timeout: opts.timeout),
          let masked = out.masked.first
    else {
        // The commonest cause on a real paste is length, and "produced nothing" reads like "found
        // no PII" — the opposite of what happened. Name both so nobody concludes their text was
        // clean when it was actually refused.
        FileHandle.standardError.write("""
            masking produced nothing — a caller would send nothing here.
            Common causes: the input is over the \(MaskerConfig.default.maxInputTokens)-token budget, \
            it exceeded --timeout, or the model failed to load.

            """.data(using: .utf8)!)
        status = 2
        return
    }
    FileHandle.standardOutput.write(masked.data(using: .utf8)!)
    if opts.showMap {
        let lines = out.restore.sorted { $0.key < $1.key }
            .map { "  \($0.key) ← \($0.value)" }
            .joined(separator: "\n")
        let summary = out.restore.isEmpty ? "  (nothing detected)" : lines
        FileHandle.standardError.write("\n\(summary)\n".data(using: .utf8)!)
    }
}
done.wait()
exit(status)

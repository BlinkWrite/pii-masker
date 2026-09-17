import Foundation
import os

/// Where the library's log lines go.
///
/// Two channels, deliberately separate. `subsystem` names an `os.Logger` for errors and notices —
/// diagnostics that carry no user text. `debug` is a sink that DOES receive raw user text: masked
/// prompts, entity counts, model paths. It is nil by default and there is no way to switch it on
/// at runtime, because a production build must never write the text it is masking to the system
/// log. A host that wants it must pass a closure, and should gate that on its own development-build
/// predicate.
public struct MaskerLogging: Sendable {
    public let subsystem: String
    public let debug: (@Sendable (String) -> Void)?

    public init(subsystem: String, debug: (@Sendable (String) -> Void)? = nil) {
        self.subsystem = subsystem
        self.debug = debug
    }

    public static let silent = MaskerLogging(subsystem: "com.github.swift-pii-masker")

    func logger(_ category: String) -> Logger {
        Logger(subsystem: subsystem, category: category)
    }

    func trace(_ message: @autoclosure () -> String) {
        guard let debug else { return }
        debug(message())
    }
}

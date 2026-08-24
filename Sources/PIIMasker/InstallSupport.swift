import CryptoKit
import Foundation

/// Two filesystem helpers the installer needs: a streaming SHA-256 of a file, and a path-component
/// safety check for a version string before it is used as a directory name.
public enum InstallSupport {
    /// `version` becomes an on-disk path component (the install version dir, the `current` symlink
    /// target), so a slash or `..` would escape the install root — turning `removeItem(at:)` into a
    /// recursive delete of an arbitrary directory. A ``ModelPin`` is public and a caller can build
    /// one, so this is checked at every filesystem use rather than assumed.
    public static func isSafePathComponent(_ v: String) -> Bool {
        v != "." && v != ".." && v.range(of: #"^[A-Za-z0-9._-]+$"#, options: .regularExpression) != nil
    }

    /// Streamed in 1 MB chunks: the archive is ~137 MB and the weights ~188 MiB, so neither is read
    /// into memory whole.
    public static func sha256(ofFileAt url: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: url)
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let chunk = handle.readData(ofLength: 1 << 20)
            if chunk.isEmpty { break }
            hasher.update(data: chunk)
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

using System;

namespace PIIMasker;

/// <summary>Where the library's log lines go.</summary>
/// <remarks>
/// Two channels, deliberately separate. <see cref="Diagnostic"/> receives errors and notices that
/// carry no user text. <see cref="Trace"/> DOES receive raw user text — masked prompts, entity
/// counts, model paths — so it is null by default and there is no way to switch it on at runtime,
/// because a production build must never write the text it is masking to a log. A host that wants
/// it passes a delegate and should gate that on its own development-build predicate.
/// </remarks>
public sealed class MaskerLogging
{
    /// <summary>Errors and notices. Never carries user text.</summary>
    public Action<string>? Diagnostic { get; init; }

    /// <summary>Raw user text. Null unless a host deliberately opts in.</summary>
    public Action<string>? Trace { get; init; }

    /// <summary>Logs nothing at all. The default for any caller that does not ask otherwise.</summary>
    public static MaskerLogging Silent { get; } = new();

    internal void Note(string message) => Diagnostic?.Invoke(message);

    /// <summary>
    /// Deferred so a disabled trace does no formatting work. The argument is user text; building it
    /// eagerly would cost on every masking call in a release build that never writes it anywhere.
    /// </summary>
    internal void TraceText(Func<string> message)
    {
        if (Trace is { } sink) sink(message());
    }
}

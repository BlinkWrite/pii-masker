using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using PIIMasker;

namespace PIIMasker.Cli;

// A small pipe-text-through-it tool, so anyone can see what the masker catches without writing a
// line of C#. Reads stdin, prints the masked text, and — with --show-map — what each placeholder
// stood for.
//
//   echo "email me at a@b.com" | pii-mask --model /path/to/gliner
//
// Deliberately prints the RESTORE MAP only on request: the whole point of the tool is to show what
// would leave the machine, and dumping the originals next to it undermines reading that at a glance.
//
// Ported from the Swift target's `pii-mask`, with the same options, the same exit statuses and the
// same wording, so a reader can compare the two targets on the same input.
internal static class Program
{
    private sealed class Options
    {
        public string? ModelDirectory { get; set; }
        public bool ShowMap { get; set; }
        public double Timeout { get; set; } = 30;
        public float? Threshold { get; set; }
    }

    /// <summary>
    /// <c>--help</c> is a successful request for the help text, so it goes to stdout and exits 0; a
    /// usage error goes to stderr and exits 1.
    /// </summary>
    /// <remarks>
    /// Conflating the two makes <c>pii-mask --help | less</c> print nothing and breaks any script
    /// that checks the status.
    /// </remarks>
    private static int Usage(bool asked)
    {
        var text = $"""
            pii-mask — pipe text through the masker and see what it catches.

            USAGE
              pii-mask [--model DIR] [--show-map] [--threshold N] [--timeout SECONDS]

              Reads text on stdin and writes the masked text to stdout.

            OPTIONS
              --model DIR      Directory holding model.onnx, tokenizer.json and
                               tokenizer_config.json. Defaults to $PII_MASKER_MODEL_DIR.
              --show-map       Also print each placeholder and the text it replaced, on stderr.
              --threshold N    Score floor for keeping a span (default {MaskerConfig.Default.Threshold}).
                               Lower catches more and false-positives more.
              --timeout N      Seconds the pass may take (default 30).

            EXIT STATUS
              0  masked
              1  bad usage, or no model
              2  the masker produced nothing — a caller would send nothing here.
                 Includes input over the token budget. Text longer than the model's
                 window is split into several passes, not dropped.

            """;
        (asked ? Console.Out : Console.Error).Write(text);
        return asked ? 0 : 1;
    }

    private static async Task<int> Main(string[] args)
    {
        var options = new Options();
        var queue = new Queue<string>(args);
        while (queue.Count > 0)
        {
            var argument = queue.Dequeue();
            switch (argument)
            {
                case "--model":
                    if (queue.Count == 0) return Usage(false);
                    options.ModelDirectory = queue.Dequeue();
                    break;
                case "--show-map":
                    options.ShowMap = true;
                    break;
                case "--threshold":
                    if (queue.Count == 0
                        || !float.TryParse(queue.Dequeue(), NumberStyles.Float,
                            CultureInfo.InvariantCulture, out var threshold))
                        return Usage(false);
                    options.Threshold = threshold;
                    break;
                case "--timeout":
                    if (queue.Count == 0
                        || !double.TryParse(queue.Dequeue(), NumberStyles.Float,
                            CultureInfo.InvariantCulture, out var seconds))
                        return Usage(false);
                    options.Timeout = seconds;
                    break;
                case "-h":
                case "--help":
                    return Usage(true);
                default:
                    Console.Error.WriteLine($"unknown argument: {argument}");
                    return Usage(false);
            }
        }

        var environmentDirectory = Environment.GetEnvironmentVariable("PII_MASKER_MODEL_DIR");
        var modelDirectory = options.ModelDirectory
            ?? (string.IsNullOrEmpty(environmentDirectory) ? null : environmentDirectory);
        if (modelDirectory == null)
        {
            Console.Error.WriteLine("no model: pass --model DIR or set PII_MASKER_MODEL_DIR");
            return 1;
        }
        if (!ModelInstaller.IsCompleteModelDir(modelDirectory))
        {
            Console.Error.WriteLine(
                $"{modelDirectory} is missing one of {string.Join(", ", ModelInstaller.RequiredModelFiles)}.");
            Console.Error.WriteLine(
                "Unpack the archive ModelPin.Current names, or fetch the loose files from the same revision.");
            return 1;
        }

        var input = await Console.In.ReadToEndAsync().ConfigureAwait(false);
        if (string.IsNullOrWhiteSpace(input)) return 0;

        var config = options.Threshold is { } floor
            ? MaskerConfig.Default with { Threshold = floor }
            : MaskerConfig.Default;
        using var masker = new PrivacyFilter(
            modelDirectory, config: config, timeout: TimeSpan.FromSeconds(options.Timeout));

        var result = await masker.MaskAsync([input]).ConfigureAwait(false);
        if (result == null || result.Fields.Count == 0)
        {
            // The commonest cause on a real paste is length, and "produced nothing" reads like
            // "found no PII" — the opposite of what happened. Name both so nobody concludes their
            // text was clean when it was actually refused.
            Console.Error.WriteLine("masking produced nothing — a caller would send nothing here.");
            Console.Error.WriteLine(
                $"Common causes: the input is over the {MaskerConfig.Default.MaxInputTokens}-token budget, "
                + "it exceeded --timeout, or the model failed to load.");
            return 2;
        }

        Console.Out.Write(result.Fields[0]);
        if (options.ShowMap)
        {
            var summary = result.Restore.Count == 0
                ? "  (nothing detected)"
                : string.Join(Environment.NewLine, result.Restore
                    .OrderBy(pair => pair.Key, StringComparer.Ordinal)
                    .Select(pair => $"  {pair.Key} ← {pair.Value}"));
            Console.Error.WriteLine();
            Console.Error.WriteLine(summary);
        }
        return 0;
    }
}

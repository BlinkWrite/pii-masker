using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Serialization;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// The pin is now stated three times: once in Swift, once in C#, and once in <c>model.json</c> for
/// a reader to diff without building anything. Three statements of one fact drift; the manifest is
/// the shared fixture that makes any drift a red build in whichever target moved.
/// </summary>
public sealed class ModelPinTests
{
    private static string RepoRoot =>
        typeof(ModelPinTests).Assembly
            .GetCustomAttributes<AssemblyMetadataAttribute>()
            .First(a => a.Key == "PIIMaskerRepoRoot").Value
        ?? throw new InvalidOperationException("The repository root was not stamped into the test assembly.");

    private sealed record PinJson(
        [property: JsonPropertyName("version")] string Version,
        [property: JsonPropertyName("sourceURL")] string SourceUrl,
        [property: JsonPropertyName("archiveSHA256")] string ArchiveSha256,
        [property: JsonPropertyName("weightsSHA256")] string WeightsSha256,
        [property: JsonPropertyName("bytes")] long Bytes,
        [property: JsonPropertyName("maxWidth")] int MaxWidth,
        [property: JsonPropertyName("maxSequenceLength")] int MaxSequenceLength,
        [property: JsonPropertyName("files")] FilesJson Files);

    private sealed record FilesJson(
        [property: JsonPropertyName("tokenizerSHA256")] string TokenizerSha256,
        [property: JsonPropertyName("tokenizerConfigSHA256")] string TokenizerConfigSha256);

    private static PinJson ReadManifest()
    {
        var path = Path.Combine(RepoRoot, "model.json");
        Assert.True(File.Exists(path), $"The canonical manifest is missing at {path}.");
        return JsonSerializer.Deserialize<PinJson>(File.ReadAllText(path))
            ?? throw new InvalidOperationException("model.json did not parse.");
    }

    /// <summary>
    /// The .NET half of the contract the Swift <c>ModelPinTests.modelJSONMatchesTheCurrentPin</c>
    /// asserts. Both read the same document, so the two targets cannot disagree about which weights
    /// they accept without one of them going red.
    /// </summary>
    [Fact]
    public void ModelJsonMatchesTheCurrentPin()
    {
        var json = ReadManifest();
        var pin = ModelPin.Current;

        Assert.Equal(json.Version, pin.Version);
        Assert.Equal(json.SourceUrl, pin.SourceUri.AbsoluteUri);
        Assert.Equal(json.ArchiveSha256, pin.ArchiveSha256);
        Assert.Equal(json.WeightsSha256, pin.WeightsSha256);
        Assert.Equal(json.Bytes, pin.Bytes);
        Assert.Equal(json.MaxWidth, pin.MaxWidth);
        Assert.Equal(json.MaxSequenceLength, pin.MaxSequenceLength);
    }

    /// <summary>
    /// The loose-file hashes have no Swift counterpart — that target never opens the files — so this
    /// half of the manifest is pinned here alone. Without it the block could rot unnoticed, and a
    /// tokenizer swap is exactly the substitution the weights hash does not catch.
    /// </summary>
    [Fact]
    public void ModelJsonMatchesTheCurrentPinsLooseFileHashes()
    {
        var files = ReadManifest().Files;

        Assert.Equal(files.TokenizerSha256, ModelPin.Current.Files.TokenizerSha256);
        Assert.Equal(files.TokenizerConfigSha256, ModelPin.Current.Files.TokenizerConfigSha256);
    }

    /// <summary>
    /// Every entry has to be installable: the version becomes a directory name, the hashes have to
    /// be hashes, and the URL has to be immutable — a <c>resolve/main</c> URL would let the bytes
    /// under a published pin change, which is the whole thing pinning exists to prevent.
    /// </summary>
    [Fact]
    public void EveryKnownPinIsWellFormed()
    {
        Assert.NotEmpty(ModelPin.Known);
        foreach (var pin in ModelPin.Known)
        {
            Assert.True(InstallSupport.IsSafePathComponent(pin.Version), pin.Version);
            AssertIsSha256(pin.ArchiveSha256, $"{pin.Version} archive hash");
            AssertIsSha256(pin.WeightsSha256, $"{pin.Version} weights hash");
            AssertIsSha256(pin.Files.TokenizerSha256, $"{pin.Version} tokenizer hash");
            AssertIsSha256(pin.Files.TokenizerConfigSha256, $"{pin.Version} tokenizer config hash");
            Assert.True(pin.Bytes > 0 && pin.Bytes < ModelInstaller.MaxPlausibleArchiveBytes,
                $"{pin.Version} bytes");
            Assert.True(pin.MaxWidth > 0, $"{pin.Version} maxWidth");
            // A window this small could not fit the label preamble, let alone any text — so a value
            // that low is a typo, not a conservative choice.
            Assert.True(pin.MaxSequenceLength > 64, $"{pin.Version} maxSequenceLength");
            Assert.DoesNotContain("/resolve/main/", pin.SourceUri.AbsoluteUri, StringComparison.Ordinal);
        }
    }

    private static void AssertIsSha256(string digest, string what)
    {
        Assert.True(digest.Length == 64, what);
        Assert.True(digest.All(Uri.IsHexDigit), what);
    }

    /// <summary>
    /// <see cref="ModelPin.Known"/> is oldest → newest, and <see cref="ModelPin.Current"/> is the
    /// newest. Rollback walks this ordering, so an entry appended in the wrong place would revert
    /// forwards — installing the very version a probe had just rejected.
    /// </summary>
    [Fact]
    public void KnownIsOrderedAndCurrentIsTheNewest()
    {
        Assert.Equal(ModelPin.Known[^1], ModelPin.Current);
        Assert.Equal(ModelPin.Known.Count, ModelPin.Known.Select(p => p.Version).Distinct().Count());
        Assert.Null(ModelPin.Predecessor(ModelPin.Known[0].Version));
        Assert.Null(ModelPin.Predecessor("never-published"));
        for (var i = 1; i < ModelPin.Known.Count; i++)
            Assert.Equal(ModelPin.Known[i - 1], ModelPin.Predecessor(ModelPin.Known[i].Version));
    }

    /// <summary>
    /// Relocation changes where bytes are fetched and nothing about which bytes are accepted. This
    /// is the property that lets a host point at its own distribution CDN, or at a local fixture,
    /// without weakening the pin — so it is asserted field by field rather than by equality, which
    /// would pass even if relocation quietly rebuilt the record.
    /// </summary>
    [Fact]
    public void RelocatingAPinCannotAlterTheIdentityItAccepts()
    {
        var original = ModelPin.Current;
        var relocated = original.WithSourceUri(new Uri("https://dl.example.test/models/gliner.tar.gz"));

        Assert.Equal("https://dl.example.test/models/gliner.tar.gz", relocated.SourceUri.AbsoluteUri);
        Assert.Equal(original.Version, relocated.Version);
        Assert.Equal(original.ArchiveSha256, relocated.ArchiveSha256);
        Assert.Equal(original.WeightsSha256, relocated.WeightsSha256);
        Assert.Equal(original.Bytes, relocated.Bytes);
        Assert.Equal(original.MaxWidth, relocated.MaxWidth);
        Assert.Equal(original.MaxSequenceLength, relocated.MaxSequenceLength);
        Assert.Equal(original.Files, relocated.Files);
        Assert.Equal(original, original.WithSourceUri(original.SourceUri));
    }

    [Fact]
    public void ARelativeSourceUriIsRefused()
    {
        Assert.Throws<ArgumentException>(
            () => ModelPin.Current.WithSourceUri(new Uri("models/gliner.tar.gz", UriKind.Relative)));
    }
}

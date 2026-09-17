using System;
using System.IO;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// Promoting into a store the other target created.
/// </summary>
/// <remarks>
/// <para>
/// The two targets promote differently — Swift points a <c>current</c> symlink at a version
/// directory, this one writes a <c>current</c> file naming it, because Windows cannot create a
/// symlink without elevation or Developer Mode. <see cref="ModelStore.ResolvePromoted"/> reads both,
/// which is what lets either target load a store the other built.
/// </para>
/// <para>
/// Writing is the half that was missing. <c>File.Move</c> will not replace a directory entry, so
/// promoting over a Swift-shaped <c>current</c> failed at the last step of an install — with a
/// downloaded, hash-verified model already on disk and nothing pointing at it. A store both targets
/// can read has to be a store both targets can write.
/// </para>
/// </remarks>
public sealed class PromotionPointerTests : IDisposable
{
    private readonly string root = Path.Combine(
        Path.GetTempPath(), "piimasker-promote-" + Guid.NewGuid().ToString("N")[..12]);

    public PromotionPointerTests()
    {
        Directory.CreateDirectory(Path.Combine(root, "2026.08.1"));
        File.WriteAllText(Path.Combine(root, "2026.08.1", "model.onnx"), "the weights");
    }

    public void Dispose()
    {
        try { Directory.Delete(root, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    /// <summary>Whether this machine lets an unprivileged process create a directory symlink.</summary>
    /// <remarks>
    /// False on a stock Windows install without Developer Mode — which is the whole reason the
    /// pointer file exists. The CI matrix includes Linux, where it is always true, so the symlink
    /// case is exercised on every run even though most Windows developers will see it skipped.
    /// </remarks>
    internal static bool SymlinksAreCreatable
    {
        get
        {
            var probe = Path.Combine(Path.GetTempPath(), "piimasker-symlink-" + Guid.NewGuid().ToString("N")[..8]);
            var target = probe + "-target";
            try
            {
                Directory.CreateDirectory(target);
                Directory.CreateSymbolicLink(probe, target);
                return true;
            }
            catch (Exception) { return false; }
            finally
            {
                try { Directory.Delete(probe); } catch (Exception) { /* best effort */ }
                try { Directory.Delete(target, recursive: true); } catch (Exception) { /* best effort */ }
            }
        }
    }

    /// <summary>Promoting over a Swift-shaped <c>current</c> works, and the version it pointed at survives.</summary>
    [SymlinkFact]
    public void ASymlinkShapedCurrentIsReplacedWithoutTouchingItsTarget()
    {
        var version = Path.Combine(root, "2026.08.1");
        Directory.CreateSymbolicLink(Path.Combine(root, "current"), version);

        ModelInstaller.PointCurrent("2026.08.1", root);

        Assert.Equal(version, ModelStore.ResolvePromoted(root));
        // Removing a symlink deletes the link, never what it points at.
        Assert.True(File.Exists(Path.Combine(version, "model.onnx")),
            "the version directory the symlink pointed at was deleted with it");
    }

    /// <summary>
    /// A real directory named <c>current</c> is refused rather than deleted.
    /// </summary>
    /// <remarks>
    /// <see cref="ModelStore.ResolvePromoted"/> accepts one as a model directory in its own right,
    /// so deleting it to make room for a pointer could destroy the model being promoted over. The
    /// refusal names the reason, because the generic "could not update the promotion pointer" sent
    /// whoever hit it looking at permissions.
    /// </remarks>
    [Fact]
    public void ARealDirectoryNamedCurrentIsRefusedRatherThanDeleted()
    {
        var current = Path.Combine(root, "current");
        Directory.CreateDirectory(current);
        File.WriteAllText(Path.Combine(current, "model.onnx"), "someone's model");

        var failure = Assert.Throws<ModelInstaller.InstallException>(
            () => ModelInstaller.PointCurrent("2026.08.1", root));

        Assert.Equal(ModelInstaller.FailureKind.StoreUnwritable, failure.Kind);
        Assert.Contains("real directory", failure.Message, StringComparison.OrdinalIgnoreCase);
        Assert.True(File.Exists(Path.Combine(current, "model.onnx")), "the directory was deleted");
    }

    /// <summary>The ordinary case: no pointer yet, then one, then a different one.</summary>
    [Fact]
    public void ThePointerIsWrittenAndThenReplaced()
    {
        Directory.CreateDirectory(Path.Combine(root, "2026.09.1"));

        ModelInstaller.PointCurrent("2026.08.1", root);
        Assert.Equal(Path.Combine(root, "2026.08.1"), ModelStore.ResolvePromoted(root));

        ModelInstaller.PointCurrent("2026.09.1", root);
        Assert.Equal(Path.Combine(root, "2026.09.1"), ModelStore.ResolvePromoted(root));
    }

    /// <summary>A version that is not a plain directory entry never reaches the filesystem.</summary>
    [Theory]
    [InlineData("..")]
    [InlineData("../elsewhere")]
    [InlineData("sub/dir")]
    [InlineData("NUL")]
    public void AnUnsafeVersionIsRefusedBeforeAnythingIsWritten(string version)
    {
        var failure = Assert.Throws<ModelInstaller.InstallException>(
            () => ModelInstaller.PointCurrent(version, root));

        Assert.Equal(ModelInstaller.FailureKind.StoreUnwritable, failure.Kind);
        Assert.Null(ModelStore.ResolvePromoted(root));
    }

    /// <summary>No temp file is left behind when the promotion is refused.</summary>
    [Fact]
    public void ARefusedPromotionLeavesNoTemporaryFile()
    {
        Directory.CreateDirectory(Path.Combine(root, "current"));

        Assert.Throws<ModelInstaller.InstallException>(
            () => ModelInstaller.PointCurrent("2026.08.1", root));

        Assert.Empty(Directory.GetFiles(root, ".current-*"));
    }
}

/// <summary>A fact that only runs where an unprivileged process can create a symlink.</summary>
internal sealed class SymlinkFactAttribute : FactAttribute
{
    public SymlinkFactAttribute()
    {
        if (!PromotionPointerTests.SymlinksAreCreatable)
            Skip = "symlinks need elevation or Developer Mode here; the CI Linux leg covers this";
    }
}

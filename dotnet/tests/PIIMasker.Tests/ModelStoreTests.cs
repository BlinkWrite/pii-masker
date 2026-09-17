using System;
using System.Collections.Generic;
using System.IO;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// Where the model is looked for, and what the loader last concluded about it.
/// </summary>
/// <remarks>
/// <para>
/// The counterpart of the Swift target's <c>ModelStoreTests</c>, plus the cases that only exist
/// here: this target accepts a <c>current</c> plain-text pointer as well as a symlink, and that
/// file is something a caller can write — so what it is allowed to name is a security boundary, not
/// a convenience.
/// </para>
/// <para>
/// The store is process-wide state, so these run in one collection and each test configures it.
/// </para>
/// </remarks>
[Collection(nameof(ModelStoreTests))]
[CollectionDefinition(nameof(ModelStoreTests), DisableParallelization = true)]
public sealed class ModelStoreTests : IDisposable
{
    private readonly string root = Path.Combine(
        Path.GetTempPath(), "piimasker-store-" + Guid.NewGuid().ToString("N")[..12]);

    public ModelStoreTests() => Directory.CreateDirectory(root);

    public void Dispose()
    {
        ModelStore.Configure(ModelStore.Location.Unconfigured);
        try { Directory.Delete(root, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
    }

    /// <summary>A directory that looks like a model to <see cref="ModelInstaller.IsCompleteModelDir"/>.</summary>
    private string CompleteDir(string name)
    {
        var directory = Path.Combine(root, name);
        Directory.CreateDirectory(directory);
        foreach (var file in ModelInstaller.RequiredModelFiles)
            File.WriteAllText(Path.Combine(directory, file), "x");
        return directory;
    }

    private void Promote(string version) =>
        File.WriteAllText(Path.Combine(root, "current"), version);

    /// <summary>
    /// A host that forgot to configure gets "no model" rather than a guess, so its fail-closed gate
    /// holds everything.
    /// </summary>
    [Fact]
    public void UnconfiguredResolvesNothing()
    {
        ModelStore.Configure(ModelStore.Location.Unconfigured);

        Assert.Null(ModelStore.ResolvedModelDirectory());
        Assert.False(ModelStore.ModelIsInstalled());
    }

    /// <summary>The install root wins, and it resolves through the promotion pointer.</summary>
    [Fact]
    public void TheInstallRootResolvesThroughThePromotionPointer()
    {
        CompleteDir("2026.08.1");
        Promote("2026.08.1");
        ModelStore.Configure(root);

        Assert.Equal(Path.Combine(root, "2026.08.1"), ModelStore.ResolvedModelDirectory());
        Assert.True(ModelStore.ModelIsInstalled());
    }

    /// <summary>
    /// Fallbacks are searched after the install root, in order, and each must be complete on its own.
    /// </summary>
    /// <remarks>
    /// Answering per file instead would let a directory missing one file still report installed by
    /// resolving that file from a different candidate — and the loader, which reads all three from
    /// one folder, would then fail with nothing to retry it.
    /// </remarks>
    [Fact]
    public void FallbacksAreOrderedAndMustEachBeComplete()
    {
        var incomplete = Path.Combine(root, "incomplete");
        Directory.CreateDirectory(incomplete);
        File.WriteAllText(Path.Combine(incomplete, ModelInstaller.RequiredModelFiles[0]), "x");
        var good = CompleteDir("shipped");
        var alsoGood = CompleteDir("checkout");

        ModelStore.Configure(new ModelStore.Location(null, [incomplete, good, alsoGood]));

        Assert.Equal(good, ModelStore.ResolvedModelDirectory());
    }

    /// <summary>
    /// A promotion pointing at an incomplete directory falls through to the fallbacks rather than
    /// resolving to something the loader cannot open.
    /// </summary>
    [Fact]
    public void AnIncompletePromotionFallsThrough()
    {
        Directory.CreateDirectory(Path.Combine(root, "2026.08.1"));
        Promote("2026.08.1");
        var shipped = CompleteDir("shipped");

        ModelStore.Configure(new ModelStore.Location(root, [shipped]));

        Assert.Equal(shipped, ModelStore.ResolvedModelDirectory());
    }

    /// <summary>Re-configuring throws away everything concluded about the previous location.</summary>
    [Fact]
    public void ReconfiguringInvalidates()
    {
        CompleteDir("2026.08.1");
        Promote("2026.08.1");
        ModelStore.Configure(root);
        Assert.True(ModelStore.ModelIsInstalled());
        ModelStore.MarkLoaded();
        Assert.Equal(ModelStore.State.Loaded, ModelStore.CurrentState);

        ModelStore.Configure(ModelStore.Location.Unconfigured);

        // The cached "installed" verdict went with it, or a relocated host would keep reporting a
        // model it can no longer find.
        Assert.False(ModelStore.ModelIsInstalled());
        Assert.Equal(ModelStore.State.Unknown, ModelStore.CurrentState);
    }

    /// <summary>A real change raises the event once; setting the same state again raises nothing.</summary>
    [Fact]
    public void StateChangesArePushedOnce()
    {
        ModelStore.Configure(ModelStore.Location.Unconfigured);
        var raised = 0;
        void Handler(object? sender, EventArgs e) => raised++;
        ModelStore.StateChanged += Handler;
        try
        {
            ModelStore.MarkLoaded();
            ModelStore.MarkLoaded();
            Assert.Equal(1, raised);

            ModelStore.MarkUnusable();
            Assert.Equal(2, raised);
        }
        finally { ModelStore.StateChanged -= Handler; }
    }

    /// <summary>
    /// Only the loader may certify. <see cref="ModelStore.Invalidate"/> returns the state to
    /// "nothing known", never to "fine".
    /// </summary>
    [Fact]
    public void InvalidatingReturnsToUnknownRatherThanHealthy()
    {
        ModelStore.MarkUnusable();
        Assert.Equal(ModelStore.State.Unusable, ModelStore.CurrentState);

        ModelStore.Invalidate();

        Assert.Equal(ModelStore.State.Unknown, ModelStore.CurrentState);
    }

    /// <summary>Every invalidation advances the generation, so a cached masker can tell it is stale.</summary>
    [Fact]
    public void InvalidationAdvancesTheGeneration()
    {
        var before = ModelStore.Generation;

        ModelStore.Invalidate();

        Assert.True(ModelStore.Generation > before,
            $"generation did not advance: {before} → {ModelStore.Generation}");
    }

    /// <summary>
    /// The promotion pointer may only name a plain directory entry inside the install root.
    /// </summary>
    /// <remarks>
    /// No Swift counterpart — Swift promotes with a symlink and has no file to validate. Here the
    /// file is on disk, so a caller able to write it must not be able to aim the loader, or a later
    /// recursive delete, at anything outside the root. Every one of these resolves to null rather
    /// than to a path.
    /// </remarks>
    [Theory]
    [InlineData("..")]
    [InlineData("../elsewhere")]
    [InlineData("..\\elsewhere")]
    [InlineData("/etc")]
    [InlineData("C:\\Windows")]
    [InlineData("sub/dir")]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("NUL")]
    [InlineData("COM1.2026")]
    public void APromotionPointerCannotAimOutsideTheInstallRoot(string named)
    {
        Promote(named);

        Assert.Null(ModelStore.ResolvePromoted(root));
    }

    /// <summary>A symlink-shaped promotion — a real <c>current</c> directory — resolves too.</summary>
    /// <remarks>
    /// That is the Swift target's layout, and a store written by either target has to resolve in
    /// both, or a user moving between them silently loses their installed model.
    /// </remarks>
    [Fact]
    public void ADirectoryNamedCurrentResolvesAsTheSwiftLayout()
    {
        var current = Path.Combine(root, "current");
        Directory.CreateDirectory(current);

        Assert.Equal(current, ModelStore.ResolvePromoted(root));
    }

    /// <summary>Nothing promoted is not an error; it is "no model yet".</summary>
    [Fact]
    public void NoPromotionResolvesToNull()
    {
        Assert.Null(ModelStore.ResolvePromoted(root));
    }

    /// <summary>Surrounding whitespace in the pointer is not a different version.</summary>
    [Fact]
    public void ThePromotionPointerIsTrimmed()
    {
        CompleteDir("2026.08.1");
        Promote("  2026.08.1\r\n");

        Assert.Equal(Path.Combine(root, "2026.08.1"), ModelStore.ResolvePromoted(root));
    }
}

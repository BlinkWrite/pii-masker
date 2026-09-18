using System;
using System.Collections.Generic;
using System.Formats.Tar;
using System.IO;
using System.IO.Compression;
using System.Linq;
using System.Net;
using System.Net.Http;
using System.Security.Cryptography;
using System.Text;
using System.Threading;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// The install pipeline over HTTP: the streaming download, the progress it reports, and what it
/// leaves behind when a transfer goes wrong.
/// </summary>
/// <remarks>
/// <para>
/// <see cref="ModelInstallerTests"/> drives the same pipeline over <c>file://</c>, which takes a
/// different branch — a single <c>File.Copy</c>. So every install test in this project used to stop
/// short of the code that actually pulls bytes from a distribution host: the read/write loop, the
/// progress arithmetic, and the classification of a failed transfer. That is the half a user's
/// first run depends on.
/// </para>
/// <para>
/// Driven through an injected <see cref="HttpMessageHandler"/> rather than a socket, so there is no
/// port to bind, no firewall to placate and nothing to time out — the same test runs identically on
/// a developer's machine and on a CI runner.
/// </para>
/// </remarks>
public sealed class ModelDownloadTests : IDisposable
{
    private readonly string scratch = Path.Combine(
        Path.GetTempPath(), "piimasker-download-" + Guid.NewGuid().ToString("N")[..12]);

    public ModelDownloadTests() => Directory.CreateDirectory(scratch);

    public void Dispose()
    {
        try { Directory.Delete(scratch, recursive: true); }
        catch (IOException) { }
        catch (UnauthorizedAccessException) { }
    }

    private string InstallRoot => Path.Combine(scratch, "store");

    private static readonly Uri Source = new("https://dl.example.test/gliner-pii.tar.gz");

    /// <summary>A real gzipped tar holding the three required files, as bytes rather than a file.</summary>
    private static byte[] ArchiveBytes(string weights = "the weights")
    {
        using var buffer = new MemoryStream();
        using (var gzip = new GZipStream(buffer, CompressionLevel.SmallestSize, leaveOpen: true))
        using (var tar = new TarWriter(gzip, TarEntryFormat.Pax))
        {
            var entries = new Dictionary<string, string>
            {
                ["model.onnx"] = weights,
                ["tokenizer.json"] = "{\"tokenizer\":true}",
                ["tokenizer_config.json"] = "{\"config\":true}",
            };
            foreach (var (name, content) in entries)
            {
                tar.WriteEntry(new PaxTarEntry(TarEntryType.RegularFile, name)
                {
                    DataStream = new MemoryStream(Encoding.UTF8.GetBytes(content)),
                });
            }
        }
        return buffer.ToArray();
    }

    private static string Sha256(byte[] bytes) =>
        Convert.ToHexString(SHA256.HashData(bytes)).ToLowerInvariant();

    private static string Sha256(string text) => Sha256(Encoding.UTF8.GetBytes(text));

    private static ModelPin PinFor(byte[] archive, string weights = "the weights") =>
        ModelPin.Current with
        {
            Version = "2026.08.1",
            SourceUri = Source,
            ArchiveSha256 = Sha256(archive),
            WeightsSha256 = Sha256(weights),
            Bytes = archive.Length,
        };

    private ModelInstaller InstallerFor(ModelPin pin, HttpMessageHandler handler) =>
        new(InstallRoot, pin, new HttpClient(handler));

    /// <summary>Serves one canned response, so a transfer can be shaped precisely.</summary>
    private sealed class StubHandler(Func<HttpResponseMessage> respond) : HttpMessageHandler
    {
        public int Calls { get; private set; }

        protected override Task<HttpResponseMessage> SendAsync(
            HttpRequestMessage request, CancellationToken cancellationToken)
        {
            Calls++;
            return Task.FromResult(respond());
        }
    }

    private static HttpResponseMessage Ok(byte[] body, bool withContentLength = true)
    {
        var content = new ByteArrayContent(body);
        if (!withContentLength) content.Headers.ContentLength = null;
        return new HttpResponseMessage(HttpStatusCode.OK) { Content = content };
    }

    /// <summary>A body that stops partway, the way a dropped connection does.</summary>
    private sealed class TruncatingStream(byte[] data, int failAfter) : Stream
    {
        private int position;

        public override int Read(byte[] buffer, int offset, int count)
        {
            if (position >= failAfter) throw new IOException("the connection was reset");
            var take = Math.Min(count, Math.Min(failAfter, data.Length) - position);
            Array.Copy(data, position, buffer, offset, take);
            position += take;
            return take;
        }

        public override bool CanRead => true;
        public override bool CanSeek => false;
        public override bool CanWrite => false;
        public override long Length => data.Length;
        public override long Position { get => position; set => throw new NotSupportedException(); }
        public override void Flush() { }
        public override long Seek(long offset, SeekOrigin origin) => throw new NotSupportedException();
        public override void SetLength(long value) => throw new NotSupportedException();
        public override void Write(byte[] buffer, int offset, int count) => throw new NotSupportedException();
    }

    [Fact]
    public async Task AnHttpDownloadInstallsAndVerifies()
    {
        var archive = ArchiveBytes();
        var installer = InstallerFor(PinFor(archive), new StubHandler(() => Ok(archive)));

        var outcome = await installer.EnsureLatestAsync();

        Assert.True(outcome.Installed);
        Assert.True(ModelInstaller.IsCompleteModelDir(outcome.Directory));
        Assert.Equal("2026.08.1", installer.InstalledVersion());
        Assert.Equal("the weights",
            await File.ReadAllTextAsync(Path.Combine(outcome.Directory, "model.onnx")));
    }

    /// <summary>
    /// Progress runs forward, stays inside 0..1, and ends with the install reported complete.
    /// </summary>
    /// <remarks>
    /// A host paints a progress bar from this. Fractions that jumped backwards, exceeded 1, or never
    /// reached it would each show up as a bar that lies — and none of it is visible from a
    /// <c>file://</c> install, which reports a single 1 and returns.
    /// </remarks>
    [Fact]
    public async Task ProgressRunsForwardAndEndsInstalled()
    {
        var archive = ArchiveBytes();
        var installer = InstallerFor(PinFor(archive), new StubHandler(() => Ok(archive)));

        var fractions = new List<double>();
        var sawInstalled = false;
        await installer.EnsureLatestAsync(onProgress: report =>
        {
            if (report is ModelInstaller.Progress.Downloading downloading)
                fractions.Add(downloading.Fraction);
            if (report is ModelInstaller.Progress.Installed) sawInstalled = true;
        });

        Assert.NotEmpty(fractions);
        Assert.All(fractions, fraction => Assert.InRange(fraction, 0d, 1d));
        Assert.Equal(fractions.OrderBy(fraction => fraction), fractions);
        Assert.Equal(1d, fractions[^1]);
        Assert.True(sawInstalled, "the install never reported itself complete");
    }

    /// <summary>A server error installs nothing and says which kind of failure it was.</summary>
    [Fact]
    public async Task AServerErrorInstallsNothing()
    {
        var archive = ArchiveBytes();
        var installer = InstallerFor(PinFor(archive), new StubHandler(
            () => new HttpResponseMessage(HttpStatusCode.InternalServerError)
            {
                Content = new ByteArrayContent([]),
            }));

        var failure = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.DownloadFailed, failure.Kind);
        Assert.Null(installer.InstalledVersion());
        AssertNothingLeftBehind();
    }

    /// <summary>
    /// A body that is not the pinned bytes is refused, however healthy the transfer looked.
    /// </summary>
    /// <remarks>
    /// The point of the pin: a distribution host that serves something else — compromised, or just
    /// wrong — cannot get it installed. Over HTTP rather than <c>file://</c> because that is the
    /// path a substituted archive would actually arrive by.
    /// </remarks>
    [Fact]
    public async Task ABodyThatIsNotThePinnedArchiveIsRefused()
    {
        var pinned = ArchiveBytes("the weights");
        var substituted = ArchiveBytes("someone else's weights");
        var installer = InstallerFor(PinFor(pinned), new StubHandler(() => Ok(substituted)));

        var failure = await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Equal(ModelInstaller.FailureKind.ChecksumMismatch, failure.Kind);
        Assert.Null(installer.InstalledVersion());
        AssertNothingLeftBehind();
    }

    /// <summary>A transfer that dies partway leaves no half-written archive to be trusted later.</summary>
    [Fact]
    public async Task AConnectionThatDiesMidDownloadLeavesNothingBehind()
    {
        var archive = ArchiveBytes();
        var installer = InstallerFor(PinFor(archive), new StubHandler(() =>
            new HttpResponseMessage(HttpStatusCode.OK)
            {
                Content = new StreamContent(new TruncatingStream(archive, archive.Length / 2)),
            }));

        await Assert.ThrowsAsync<ModelInstaller.InstallException>(
            () => installer.EnsureLatestAsync());

        Assert.Null(installer.InstalledVersion());
        AssertNothingLeftBehind();
    }

    /// <summary>
    /// A response with no Content-Length still installs, falling back to the pinned size for progress.
    /// </summary>
    /// <remarks>
    /// Chunked transfer encoding has no length, and a host that served the model that way would
    /// otherwise divide by zero or report nothing at all.
    /// </remarks>
    [Fact]
    public async Task AResponseWithoutContentLengthStillInstalls()
    {
        var archive = ArchiveBytes();
        var installer = InstallerFor(
            PinFor(archive), new StubHandler(() => Ok(archive, withContentLength: false)));

        var fractions = new List<double>();
        var outcome = await installer.EnsureLatestAsync(onProgress: report =>
        {
            if (report is ModelInstaller.Progress.Downloading downloading)
                fractions.Add(downloading.Fraction);
        });

        Assert.True(outcome.Installed);
        Assert.All(fractions, fraction => Assert.InRange(fraction, 0d, 1d));
    }

    /// <summary>A second run over HTTP re-uses what is installed rather than fetching again.</summary>
    /// <remarks>
    /// The check that keeps every launch from pulling 137 MB. Asserted on the handler's call count,
    /// because "installed == false" alone would also be true if it downloaded and discarded.
    /// </remarks>
    [Fact]
    public async Task ASecondRunDoesNotFetchAgain()
    {
        var archive = ArchiveBytes();
        var handler = new StubHandler(() => Ok(archive));
        var installer = InstallerFor(PinFor(archive), handler);

        Assert.True((await installer.EnsureLatestAsync()).Installed);
        Assert.Equal(1, handler.Calls);

        var second = await installer.EnsureLatestAsync();

        Assert.False(second.Installed);
        Assert.Equal(1, handler.Calls);
    }

    /// <summary>No archive, no staging directory, no half-written version directory.</summary>
    private void AssertNothingLeftBehind()
    {
        if (!Directory.Exists(InstallRoot)) return;
        var leftovers = Directory.EnumerateFileSystemEntries(InstallRoot, "*", SearchOption.AllDirectories)
            .Where(path => !Path.GetFileName(path).Equals("current", StringComparison.Ordinal))
            .ToList();
        Assert.True(leftovers.Count == 0,
            "the install root is not clean: " + string.Join(", ", leftovers));
    }
}

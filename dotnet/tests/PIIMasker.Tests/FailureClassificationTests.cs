using System;
using System.IO;
using System.Net.Http;
using System.Net.Sockets;
using System.Threading.Tasks;
using Xunit;

namespace PIIMasker.Tests;

/// <summary>
/// Which transport failures mean "this device has no network" and which mean "we could not reach
/// the publisher".
/// </summary>
/// <remarks>
/// The distinction is the whole reason <see cref="ModelInstaller.FailureKind.Offline"/> exists
/// separately from <see cref="ModelInstaller.FailureKind.DownloadFailed"/>: the first is the user's
/// problem to fix and the second is ours. Getting it wrong tells someone whose internet is working
/// perfectly to go and check their router.
///
/// The Swift target states the rule directly: offline is true only for codes meaning the device has
/// no usable network — Wi-Fi off, airplane mode, a dropped connection — and NOT for a reachable
/// network that cannot reach the model host, which covers a host being down, a DNS miss, a timeout
/// and a refused connection.
/// </remarks>
public sealed class FailureClassificationTests
{
    /// <summary>
    /// A DNS miss is NOT offline. Found live: an installer stamped with a hostname that does not
    /// resolve reported "No internet connection" on a machine with working internet. A lapsed
    /// domain, a misconfigured CDN record or a corporate DNS that blocks our host all land here,
    /// and all of them are the publisher's problem presented as the user's.
    /// </summary>
    [Fact]
    public void ADnsMissIsNotOffline()
    {
        var dnsMiss = new HttpRequestException(HttpRequestError.NameResolutionError, "no such host");

        Assert.False(ModelInstaller.IsOfflineError(dnsMiss));
        Assert.Equal(ModelInstaller.FailureKind.DownloadFailed, ModelInstaller.ClassifyFailure(dnsMiss));
    }

    /// <summary>A host that is simply down is the publisher's problem too.</summary>
    [Fact]
    public void AnUnreachableHostIsNotOffline()
    {
        var refused = new HttpRequestException(
            HttpRequestError.ConnectionError, "refused", new SocketException((int)SocketError.ConnectionRefused));

        Assert.False(ModelInstaller.IsOfflineError(refused));
        Assert.Equal(ModelInstaller.FailureKind.DownloadFailed, ModelInstaller.ClassifyFailure(refused));
    }

    /// <summary>A timeout is not offline either — the network answered, slowly or not at all.</summary>
    [Fact]
    public void ATimeoutIsNotOffline()
    {
        var timeout = new TaskCanceledException("timed out", new TimeoutException());

        Assert.False(ModelInstaller.IsOfflineError(timeout));
    }

    /// <summary>
    /// What IS offline: the adapter reporting the device has no network. These are the only cases
    /// where "check your connection" is the right instruction.
    /// </summary>
    [Theory]
    [InlineData(SocketError.NetworkDown)]
    [InlineData(SocketError.NetworkUnreachable)]
    public void ADeviceWithNoNetworkIsOffline(SocketError code)
    {
        var down = new HttpRequestException(
            HttpRequestError.ConnectionError, "no network", new SocketException((int)code));

        Assert.True(ModelInstaller.IsOfflineError(down));
        Assert.Equal(ModelInstaller.FailureKind.Offline, ModelInstaller.ClassifyFailure(down));
    }

    /// <summary>
    /// A host with no route to it is NOT the device being offline — it is one host being
    /// unreachable, which is the publisher's problem. Listed separately from the DNS case because
    /// the two arrive as different errors and both used to be misread as offline.
    /// </summary>
    [Theory]
    [InlineData(SocketError.HostUnreachable)]
    [InlineData(SocketError.HostNotFound)]
    [InlineData(SocketError.ConnectionRefused)]
    [InlineData(SocketError.TimedOut)]
    public void AHostWeCannotReachIsNotTheDeviceBeingOffline(SocketError code)
    {
        var unreachable = new HttpRequestException(
            HttpRequestError.ConnectionError, "cannot reach host", new SocketException((int)code));

        Assert.False(ModelInstaller.IsOfflineError(unreachable));
        Assert.Equal(ModelInstaller.FailureKind.DownloadFailed, ModelInstaller.ClassifyFailure(unreachable));
    }

    /// <summary>A full disk and an unwritable store outrank any transport reading.</summary>
    [Fact]
    public void DiskFailuresAreClassifiedAheadOfTransportOnes()
    {
        var full = new IOException("disk full") { HResult = unchecked((int)0x80070070) };
        Assert.Equal(ModelInstaller.FailureKind.OutOfSpace, ModelInstaller.ClassifyFailure(full));

        var denied = new UnauthorizedAccessException("read only");
        Assert.Equal(ModelInstaller.FailureKind.StoreUnwritable, ModelInstaller.ClassifyFailure(denied));
    }
}

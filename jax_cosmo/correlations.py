import jax
import jax.numpy as jnp

from jax_cosmo.scipy.special import loggamma_vector as loggamma


def windowfn(x, dlnxleft=0.46, dlnxright=0.46):
    xmin = x.min()
    xmax = x.max()
    xleft = jnp.exp(jnp.log(xmin) + dlnxleft)
    xright = jnp.exp(jnp.log(xmax) - dlnxright)
    w = jnp.zeros_like(x)
    indices = jnp.logical_and(x > xleft, x < xright).nonzero()
    w = jax.ops.index_update(w, indices, 1.0)

    il_indices = jnp.logical_and((x < xleft), (x > xmin)).nonzero()
    ir_indices = jnp.logical_and((x > xright), (x < xmax)).nonzero()

    rl = (x[il_indices] - xmin) / (xleft - xmin)
    rr = (xmax - x[ir_indices]) / (xmax - xright)
    w = jax.ops.index_update(
        w, il_indices, rl - jnp.sin(jnp.pi * 2 * rl) / (2 * jnp.pi)
    )
    w = jax.ops.index_update(
        w, ir_indices, rr - jnp.sin(jnp.pi * 2 * rr) / (2 * jnp.pi)
    )

    return w


def calc_Mellnu(tt, alpha, q=0):
    n = q - 1 - 1j * tt
    intjlttn = (
        2 ** (n - 1)
        * jnp.sqrt(jnp.pi)
        * jnp.exp(loggamma((1 + n) / 2.0) - loggamma((2 - n) / 2.0))
    )
    A = alpha ** (1j * tt - q)
    return A * intjlttn


def calc_phi(pk, k0, N, L, q):
    k = k0 * jnp.exp(jnp.arange(0, N) * 2 * jnp.pi / L)
    P = pk(k)
    kpk = (k / k0) ** (3 - q) * P * windowfn(k)
    phi = jnp.conj(jnp.fft.rfft(kpk)) / L
    phi *= windowfn(k)[k.shape[0] - phi.shape[0] :]
    return phi


def xicalc(pk, N=1024, kmin=1e-4, kmax=1e2, r0=1e-4):

    r"""Computes the two-point correlation given a power spectrum.
    Uses the FFTLog algorithm in (ADD REFERENCE). Original
    implementation by Cheng Zhao.

    Parameters
    ----------
    pk: function
        Power spectrum to be transformed
    N: int, optional
        Number of bins for FFT
    kmin: float, optional
        Minimum k (wavenumber) in h / Mpc
    kmax: float, optional
        Maximum k (wavenumber) in h / Mpc
    r0: float, optional
        Minimum separation in Mpc / h
    Returns
    -------
    s: array_like
        Separations in [Mpc / h]
    xi: array_like
        Correlation function monopole from
        the transformed pk

    References
    -------

    FFTLog paper: https://jila.colorado.edu/~ajsh/FFTLog/fftlog.pdf
    """

    qnu = 1.95
    N2 = int(N / 2) + 1
    k0 = kmin
    G = jnp.log(kmax / kmin)
    alpha = k0 * r0
    L = 2 * jnp.pi * N / G

    tt = jnp.arange(0, N2) * 2 * jnp.pi / G  # eq. 18 2pi * m / L
    rr = r0 * jnp.exp(jnp.arange(0, N) * (G / N))  # eq 13.
    prefac = k0 ** 3 / (jnp.pi * G) * (rr / r0) ** (-qnu)

    Mellnu = calc_Mellnu(tt, alpha, qnu)
    phi = calc_phi(pk, k0, N, L, qnu)
    print(len(rr), len(Mellnu))

    xi = prefac * jnp.fft.irfft(phi * Mellnu, N) * N
    return rr, xi


def P2xi(k, P):

    r"""Computes the two-point correlation given a power spectrum.
    Uses the FFTLog algorithm in (ADD REFERENCE)

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1}
    P: array_like
        Power spectrum to transform

    Returns
    -------
    s: array_like
        Separations in [Mpc / h]
    xi: array_like
        Correlation function monopole from
        the transformed P
    """

    Pm = P
    Pkfn = lambda x: jnp.exp(jnp.interp(jnp.log(x), jnp.log(k), jnp.log(Pm)))
    s0, xi0 = xicalc(Pkfn, k.shape[0], k.min(), k.max(), 1e-0)
    return s0, xi0


@jax.jit
def xicalc_trapz(k, P, smooth_a, sarr):

    r"""Computes the two-point correlation given a power spectrum.
    Uses integration wuth a damping factor smooth_a (see Xu et al. 2012)

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1} must be log-spaced
    P: array_like
        Power spectrum to transform evaluated at `k`
    smooth_a: float
        Smoothing scale to avoid oscillation
        issues in the integration.
    s_arr: array_like
        Array of separations at which to compute
        the correlation function

    Returns
    -------
    s_arr: array_like
        Separations in [Mpc / h]
    xi: array_like
        Correlation function monopole from
        the transformed P
    """

    P *= k ** 3 * jnp.exp(-(k ** 2) * smooth_a ** 2) * 0.5 / jnp.pi ** 2
    delta = jnp.log(k[1]) - jnp.log(k[0])

    ks = sarr[:, None] * k[None, :]
    j0 = jnp.sin(ks) / ks
    # j0 = jnp.sinc(ks)
    j2 = -(3.0 / ks ** 2 - 1.0) * j0 + 3.0 * jnp.cos(ks) / ks ** 2
    j4 = (5 * (2 * ks ** 2 - 21.0) * jnp.cos(ks)) / ks ** 4 + (
        (ks ** 4 - 45 * ks ** 2 + 105.0) * jnp.sin(ks)
    ) / ks ** 5

    xi0 = (P[None, :] * j0 * delta).sum(axis=1)
    xi2 = (P[None, :] * j2 * delta).sum(axis=1)
    xi4 = (P[None, :] * j4 * delta).sum(axis=1)

    return sarr, xi0, xi2, xi4


@jax.jit
def xicalc_trapz_linear(k, P, smooth_a, sarr):

    r"""Computes the two-point correlation given a power spectrum.
    Uses integration wuth a damping factor smooth_a (see Xu et al. 2012)

    Parameters
    ----------
    k: array_like
        Wave number in h Mpc^{-1} must be log-spaced
    P: array_like
        Power spectrum to transform evaluated at `k`
    smooth_a: float
        Smoothing scale to avoid oscillation
        issues in the integration.
    s_arr: array_like
        Array of separations at which to compute
        the correlation function

    Returns
    -------
    s_arr: array_like
        Separations in [Mpc / h]
    xi: array_like
        Correlation function monopole from
        the transformed P
    """

    P *= k ** 2 * jnp.exp(-(k ** 2) * smooth_a ** 2) * 0.5 / jnp.pi ** 2
    delta = k[1] - k[0]

    ks = sarr[:, None] * k[None, :]
    j0 = jnp.sin(ks) / ks
    j2 = -(3.0 / ks ** 2 - 1.0) * j0 + 3.0 * jnp.cos(ks) / ks ** 2
    j4 = (5 * (2 * ks ** 2 - 21.0) * jnp.cos(ks)) / ks ** 4 + (
        (ks ** 4 - 45 * ks ** 2 + 105.0) * jnp.sin(ks)
    ) / ks ** 5

    xi0 = (P[None, :] * j0 * delta).sum(axis=1)
    xi2 = (P[None, :] * j2 * delta).sum(axis=1)
    xi4 = (P[None, :] * j4 * delta).sum(axis=1)

    return sarr, xi0, xi2, xi4

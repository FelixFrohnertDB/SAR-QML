"""Dyadic (register-indexed) DDR model for SAR raw data.

Construction
------------
`n` qubits per axis index a ``2^n x 2^n`` raw block, i.e. ``4^n`` raw samples on
``2n`` qubits -- an exponential compression, which is why a task on a very large
raw image can still be simple.  The register value *is* the raw sample index.

The SAR range-azimuth (squint / RCM) coupling over that block is

    P(a,d) = exp(i * gamma * a * d),        a,d in [0, 2^n)

and expanding the indices in bits, ``a = sum_k 2^{n-1-k} a_k`` with
``a_k = (1 - Z_k)/2``, gives the *exact* operator identity

    exp(i gamma a d) = prod_{k,l} RZZ_{k,l}( gamma * sigma_kl / 4 ) x (1q phases),
    sigma_kl = 2^{(n-1-k) + (n-1-l)}                                        (1)

so the geometric coupling is exactly one DDR query layer with a **dyadic**
generator ramp and the single angle ``f = gamma``.  `check_identity` verifies (1)
to machine precision.  This is the sense in which the acquisition geometry
*determines* the query angle: there is no hyperparameter to tune.

Why it is sharp
---------------
A single point target has ``sigma_tilde(a,d) = exp(-i(a u + d v))``, a product
state over all ``2n`` bits.  The raw block is ``sigma_tilde * P``, whose Schmidt
rank across the azimuth/range cut is 1 **iff** the residual coupling
``Delta = gamma - f`` satisfies ``Delta = 0 mod 2pi``: absorbing
``exp(i Delta a d) = prod_{k,l} exp(i Delta sigma_kl a_k d_l)`` into a product
state would require ``Delta * sigma_kl = 0 mod 2pi`` for every pair, and the
smallest ramp value is ``sigma = 1``.  So the coupling *is* the entanglement of
the raw block, and only the geometric angle removes it.
"""
from __future__ import annotations
import numpy as np


# ---------------------------------------------------------------- exact identity
def _z_diag(n_q: int, q: int) -> np.ndarray:
    z = np.array([1.0, -1.0])
    o = np.ones(1)
    for k in range(n_q):
        o = np.kron(o, z if k == q else np.ones(2))
    return o


def dyadic_ramp(n: int) -> np.ndarray:
    """sigma_kl = 2^{(n-1-k)+(n-1-l)} in (k,l) row-major order."""
    return np.array([2.0 ** ((n - 1 - k) + (n - 1 - l))
                     for k in range(n) for l in range(n)])


def check_identity(n: int, gamma: float) -> float:
    """Max abs error between the RZZ product (1) and exp(i*gamma*a*d)."""
    N = 2 * n
    D = np.ones(2 ** N, dtype=complex)
    for k in range(n):
        for l in range(n):
            s = 2.0 ** ((n - 1 - k) + (n - 1 - l))
            zk, zl = _z_diag(N, k), _z_diag(N, n + l)
            D *= np.exp(1j * gamma * s / 4.0 * (1.0 - zk - zl + zk * zl))
    a = np.arange(2 ** n)
    return float(np.abs(D - np.exp(1j * gamma * np.outer(a, a)).ravel()).max())


# ---------------------------------------------------------------------- physics
def coupling(n: int, f: float) -> np.ndarray:
    a = np.arange(2 ** n)
    return np.exp(1j * f * np.outer(a, a))


def scene(n: int, u: float, v: float) -> np.ndarray:
    """Single point target at image position (u,v) -> separable product state."""
    a = np.arange(2 ** n)
    return np.outer(np.exp(-1j * a * u), np.exp(-1j * a * v))


def raw_block(n: int, u: float, v: float, gamma: float) -> np.ndarray:
    return scene(n, u, v) * coupling(n, gamma)


def schmidt(S: np.ndarray, tol: float = 1e-8):
    s = np.linalg.svd(S, compute_uv=False)
    s = s / np.linalg.norm(s)
    p = s ** 2
    p = p[p > 1e-15]
    return int((p > tol).sum()), float(-(p * np.log2(p)).sum())


def design(n: int, X: np.ndarray) -> np.ndarray:
    """E[i, (a,d)] = exp(-i (a x1 + d x2)) -- image-domain readout (Hadamard test)."""
    a = np.arange(2 ** n)
    Ea = np.exp(-1j * X[:, 0:1] * a[None, :])
    Ed = np.exp(-1j * X[:, 1:2] * a[None, :])
    return (Ea[:, :, None] * Ed[:, None, :]).reshape(len(X), 4 ** n)


def target_fn(n: int, X: np.ndarray, u: float, v: float, gamma: float,
              scale: float | None = None):
    """g*(x) = Re sum_{a,d} s(a,d) e^{-i(a x1 + d x2)}, s = raw block."""
    g = np.real(design(n, X) @ raw_block(n, u, v, gamma).ravel())
    if scale is None:
        scale = 0.9 / np.abs(g).max()
    return g * scale, scale

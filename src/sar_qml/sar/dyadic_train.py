"""Training for the dyadic SAR model.

Three arms, all with the SAME trainable circuit and hence the same parameter
count; they differ only in the query layer:

  ddr_geo   query layer at f = gamma, read off the acquisition geometry (eq. 1
            of `dyadic`).  No tuning, no model selection.
  ddr_tuned query layer at f selected on TRAINING loss over a grid -- included to
            show that tuning recovers gamma, not to compete.
  sdr       no query layer (f = 0): standard reuploading, which must build the
            geometric coupling out of its own entangling gates.

Trainable circuit: `depth` layers of (RY, RZ on every qubit) + CZ ring, then a
final (RY, RZ) layer, plus a global real scale and phase.  Parameters:
4*n*(depth+1) + 2 for 2n qubits.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import jax
jax.config.update("jax_enable_x64", True)   # complex128 throughout; without this
# JAX silently truncates to complex64.  Verified not to change any structural
# conclusion (identifiability, contrast, rank-1 floor) -- only the numerical
# floor level.  See experiments/sar/audit_dyadic.py, check M10.
import jax.numpy as jnp
import optax


def n_zz_pairs(n: int) -> int:
    return n * n


def n_params_zz(n: int, depth: int) -> int:
    """Trainable-ZZ ansatz: product layers PLUS a free angle on every
    (azimuth bit, range bit) pair.  This ansatz CAN represent the geometric
    phase screen exactly (set theta_kl = gamma*sigma_kl/4), so it is the fair
    expressivity baseline: it differs from DDR only in having to LEARN the n^2
    couplings from data instead of being handed the single geometric angle."""
    return n_params(n, depth) + n_zz_pairs(n)


def n_params(n: int, depth: int) -> int:
    """Product (separable) ansatz: 2 angles per qubit per layer, no entangling
    gates.  The trainable state is therefore rank-1 across the azimuth/range cut
    by construction, so ALL azimuth-range coupling must come from the query
    layer.  This is the whole point: it makes f identifiable."""
    return 4 * n * (depth + 1) + 2


def _apply_1q(psi, n_q, q, U):
    psi = psi.reshape((2 ** q, 2, 2 ** (n_q - q - 1)))
    return jnp.einsum("ij,ajb->aib", U, psi).reshape(-1)


def _ry(t):
    c, s = jnp.cos(t / 2), jnp.sin(t / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)


def _rz(t):
    return jnp.array([[jnp.exp(-0.5j * t), 0], [0, jnp.exp(0.5j * t)]],
                     dtype=jnp.complex128)


def _cz_ring_diag(n_q):
    idx = jnp.arange(2 ** n_q)
    d = jnp.ones(2 ** n_q)
    for q in range(n_q):
        r = (q + 1) % n_q
        bq = (idx >> (n_q - 1 - q)) & 1
        br = (idx >> (n_q - 1 - r)) & 1
        d = d * jnp.where(bq * br == 1, -1.0, 1.0)
    return d.astype(jnp.complex128)


def _zz_screen(phi, n):
    """diag(exp(-i/2 * sum_kl phi_kl Z_k Z_{n+l})) as a (2^n,2^n) block."""
    n_q = 2 * n
    idx = jnp.arange(2 ** n_q)
    acc = jnp.zeros(2 ** n_q)
    for k in range(n):
        for l in range(n):
            zk = 1.0 - 2.0 * ((idx >> (n_q - 1 - k)) & 1)
            zl = 1.0 - 2.0 * ((idx >> (n_q - 1 - (n + l))) & 1)
            acc = acc + phi[k * n + l] * zk * zl
    return jnp.exp(-0.5j * acc)


def amplitudes(theta, n, depth, entangle=False):
    """Trainable state on 2n qubits -> (2^n, 2^n) complex block.

    entangle=False (default): product ansatz, rank 1 by construction.
    entangle=True: adds a CZ ring, giving a conventional reuploading-style
    trainable circuit that CAN build azimuth-range coupling itself (used only as
    a fairness control -- see `audit_dyadic`)."""
    n_q = 2 * n
    psi = jnp.zeros(2 ** n_q, dtype=jnp.complex128).at[0].set(1.0)
    czd = _cz_ring_diag(n_q) if entangle else None
    k = 0
    for _ in range(depth):
        for q in range(n_q):
            psi = _apply_1q(psi, n_q, q, _ry(theta[k])); k += 1
            psi = _apply_1q(psi, n_q, q, _rz(theta[k])); k += 1
        if entangle:
            psi = psi * czd
    for q in range(n_q):
        psi = _apply_1q(psi, n_q, q, _ry(theta[k])); k += 1
        psi = _apply_1q(psi, n_q, q, _rz(theta[k])); k += 1
    return psi.reshape(2 ** n, 2 ** n)


def predict(params, n, depth, E, Pf, entangle=False, zz=False):
    """E: design matrix (n_pts, 4^n).  Pf: flattened query-layer phase screen.

    zz=True appends n^2 trainable ZZ angles after the rotation angles (the fair
    expressivity baseline; see `n_params_zz`)."""
    if zz:
        nz = n_zz_pairs(n)
        theta, phi = params[:-(2 + nz)], params[-(2 + nz):-2]
        scale, phase = params[-2], params[-1]
        w = (amplitudes(theta, n, depth, entangle).reshape(-1)
             * _zz_screen(phi, n) * Pf)
    else:
        theta, scale, phase = params[:-2], params[-2], params[-1]
        w = amplitudes(theta, n, depth, entangle).reshape(-1) * Pf
    z = E @ w
    return jnp.real(z * scale * jnp.exp(1j * phase))


@dataclass
class FitResult:
    params: np.ndarray
    train_mse: float
    restarts: np.ndarray


def fit(n, depth, E, y, Pf, seed=0, n_restarts=8, steps=1500, lr=0.05,
        entangle=False, zz=False):
    """Adam on MSE, best of n_restarts by final training loss."""
    E = jnp.asarray(E); y = jnp.asarray(y); Pf = jnp.asarray(Pf)
    npar = n_params_zz(n, depth) if zz else n_params(n, depth)

    def loss(p):
        return jnp.mean((predict(p, n, depth, E, Pf, entangle, zz) - y) ** 2)

    lg = jax.jit(jax.value_and_grad(loss))
    opt = optax.adam(lr)

    def one(p0):
        st = opt.init(p0)

        def body(c, _):
            p, st = c
            v, g = lg(p)
            up, st = opt.update(g, st)
            return (p + up, st), v
        (p, _), hist = jax.lax.scan(body, (p0, st), None, length=steps)
        return p, loss(p)

    one = jax.jit(one)
    rng = np.random.default_rng(seed)
    best, bl, alls = None, np.inf, []
    for _ in range(n_restarts):
        p0 = jnp.asarray(rng.uniform(-np.pi, np.pi, npar))
        p0 = p0.at[-2].set(1.0).at[-1].set(0.0)
        p, l = one(p0)
        l = float(l); alls.append(l)
        if l < bl:
            best, bl = np.asarray(p), l
    return FitResult(best, bl, np.array(alls))

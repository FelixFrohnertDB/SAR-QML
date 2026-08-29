"""Exact 2-qubit simulator for the DDR / standard-reuploading comparison.

Three circuit families share an identical trainable-parameter layout so that the
only difference between them is *where* the non-trainable structure sits:

    DDR  block l:  E_x(x) -> A_l -> CZ -> RZZ(sigma_l f) -> B_l
    SDR  block l:  E_x(x) -> A_l -> CZ -> E_x(x)          -> B_l
    B2   block l:  E_x(x) -> A_l -> CZ -> I               -> B_l   ( = DDR at f=0 )

with

    E_x(x)      = RY(x_1) (x) RY(x_2)
    A_l, B_l    in SU(2) (x) SU(2), Euler ZYZ on each qubit (3 angles / qubit)
    RZZ(phi)    = exp(-i phi Z_0 Z_1 / 2)

Initial state |00>, observable <Z_0>.  Parameter count is 12*T for every family.

Conventions
-----------
State is a length-4 complex vector, reshaped to (2, 2) with axis 0 = qubit 0
(the measured qubit) and axis 1 = qubit 1.  Basis index = 2*b_0 + b_1.

Everything is pure JAX (float64 enabled) and jit/vmap friendly: batching over
data points *and* over restarts is done with nested vmap by the callers.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from functools import partial
from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

CDTYPE = jnp.complex128
RDTYPE = jnp.float64

# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #


def ry(theta):
    """Single-qubit RY(theta) = exp(-i theta Y / 2)."""
    c = jnp.cos(theta / 2.0)
    s = jnp.sin(theta / 2.0)
    return jnp.array([[c, -s], [s, c]], dtype=CDTYPE)


def rz(theta):
    """Single-qubit RZ(theta) = exp(-i theta Z / 2)."""
    e = jnp.exp(-0.5j * theta)
    z = jnp.zeros((), dtype=CDTYPE)
    return jnp.array([[e, z], [z, jnp.conj(e)]], dtype=CDTYPE)


def su2_zyz(angles):
    """Euler ZYZ parameterisation of SU(2): RZ(c) RY(b) RZ(a) (a applied first)."""
    a, b, c = angles[0], angles[1], angles[2]
    return rz(c) @ ry(b) @ rz(a)


def apply_1q(state, u, wire: int):
    """Apply a 2x2 unitary to `wire` of a (2, 2) state."""
    if wire == 0:
        return jnp.einsum("ij,jb->ib", u, state)
    return jnp.einsum("ij,aj->ai", u, state)


# diagonal two-qubit gates, expressed as elementwise (2, 2) phase masks
_CZ_MASK = jnp.array([[1.0, 1.0], [1.0, -1.0]], dtype=CDTYPE)


def apply_cz(state):
    return state * _CZ_MASK


def rzz_mask(phi):
    """Phase mask of RZZ(phi) = exp(-i phi Z0Z1 / 2); z = +1 for |0>, -1 for |1>."""
    m = jnp.exp(-0.5j * phi)
    p = jnp.exp(+0.5j * phi)
    return jnp.array([[m, p], [p, m]], dtype=CDTYPE)


def apply_rzz(state, phi):
    return state * rzz_mask(phi)


def apply_ex(state, x):
    """E_x(x) = RY(x_1) on qubit 0, RY(x_2) on qubit 1."""
    state = apply_1q(state, ry(x[0]), 0)
    state = apply_1q(state, ry(x[1]), 1)
    return state


def apply_block(state, block_angles):
    """A_l / B_l: independent ZYZ Euler rotation on each qubit. angles: (2, 3)."""
    state = apply_1q(state, su2_zyz(block_angles[0]), 0)
    state = apply_1q(state, su2_zyz(block_angles[1]), 1)
    return state


# --------------------------------------------------------------------------- #
# architectures
# --------------------------------------------------------------------------- #

ARMS = ("ddr", "sdr", "b2")


@dataclass(frozen=True)
class Arch:
    """A circuit family.  `kind` in {'ddr', 'sdr', 'b2'}; `T` = number of blocks."""

    kind: str
    T: int = 2

    def __post_init__(self):
        if self.kind not in ARMS:
            raise ValueError(f"unknown arm {self.kind!r}; expected one of {ARMS}")
        if self.T < 1:
            raise ValueError("T must be >= 1")

    # ---- bookkeeping -----------------------------------------------------
    @property
    def param_shape(self):
        """(layer, block A/B, qubit, euler angle)."""
        return (self.T, 2, 2, 3)

    @property
    def n_params(self) -> int:
        return int(np.prod(self.param_shape))

    @property
    def sigmas(self):
        """Query-gate scalings sigma_l = l  (sigma_1 = 1, sigma_2 = 2, ...)."""
        return jnp.arange(1, self.T + 1, dtype=RDTYPE)

    @property
    def n_x_uploads(self) -> int:
        return self.T if self.kind in ("ddr", "b2") else 2 * self.T

    @property
    def n_query_gates(self) -> int:
        return self.T if self.kind == "ddr" else 0

    @property
    def uses_f(self) -> bool:
        return self.kind == "ddr"

    def summary(self) -> dict:
        return {
            "arm": self.kind,
            "T": self.T,
            "qubits": 2,
            "trainable_blocks": 2 * self.T,
            "n_params": self.n_params,
            "cz_gates": self.T,
            "x_uploads": self.n_x_uploads,
            "query_gates": self.n_query_gates,
            "encoding_slots": 2 * self.T,
        }

    # ---- forward ---------------------------------------------------------
    def expval(self, params, x, f):
        """<Z_0> for a single data point.  params: flat (n_params,), x: (2,)."""
        p = params.reshape(self.param_shape)
        state = jnp.zeros((2, 2), dtype=CDTYPE).at[0, 0].set(1.0)
        sig = self.sigmas
        for l in range(self.T):
            state = apply_ex(state, x)
            state = apply_block(state, p[l, 0])
            state = apply_cz(state)
            if self.kind == "ddr":
                state = apply_rzz(state, sig[l] * f)
            elif self.kind == "sdr":
                state = apply_ex(state, x)
            # 'b2': identity in the query slot
            state = apply_block(state, p[l, 1])
        probs = jnp.abs(state) ** 2
        # <Z_0> = P(q0 = 0) - P(q0 = 1)
        return jnp.sum(probs[0]) - jnp.sum(probs[1])

    # ---- batched / compiled ---------------------------------------------
    def predict_fn(self):
        """-> g(params, X, f) with X: (N, 2) -> (N,).

        Cached per Arch: building the jit wrapper fresh on every call forces a
        re-trace each time, which dominated the per-cell cost when generating
        data for many teachers in a Python loop.
        """
        return _predict_fn_cached(self)

    def predict(self, params, X, f=0.0):
        return self.predict_fn()(jnp.asarray(params), jnp.asarray(X),
                                 jnp.asarray(f, RDTYPE))

    def predict_many_fn(self):
        """-> g(params, X, f) with params (M, P), X (M, N, 2), f (M,) -> (M, N).

        One compiled call for all teachers, instead of M traced calls.
        """
        return _predict_many_cached(self)

    def predict_many(self, params, X, f):
        M = jnp.asarray(params).shape[0]
        return self.predict_many_fn()(
            jnp.asarray(params), jnp.asarray(X),
            jnp.broadcast_to(jnp.asarray(f, RDTYPE), (M,)))


@functools.lru_cache(maxsize=None)
def _predict_fn_cached(arch: "Arch"):
    return jax.jit(jax.vmap(arch.expval, in_axes=(None, 0, None)))


@functools.lru_cache(maxsize=None)
def _predict_many_cached(arch: "Arch"):
    per_teacher = jax.vmap(arch.expval, in_axes=(None, 0, None))   # over N
    return jax.jit(jax.vmap(per_teacher, in_axes=(0, 0, 0)))       # over M


# --------------------------------------------------------------------------- #
# broadcast-batched forward pass
# --------------------------------------------------------------------------- #
#
# The nested-vmap path above rebuilds every gate for each (problem, restart,
# data point) triple, although the encoding gates depend only on the data point
# and the trainable blocks only on the parameters.  The routine below builds each
# gate once on its own axis and lets broadcasting do the rest, which is what
# makes the sweeps affordable.  `expval_batched` is checked against `expval` to
# machine precision in tests/test_ddr2q.py.


def _ry_b(theta):
    """RY on a leading batch: theta (...,) -> (..., 2, 2)."""
    c = jnp.cos(theta / 2.0).astype(CDTYPE)
    s = jnp.sin(theta / 2.0).astype(CDTYPE)
    return jnp.stack([jnp.stack([c, -s], -1), jnp.stack([s, c], -1)], -2)


def _su2_b(angles):
    """ZYZ Euler on a leading batch: angles (..., 3) -> (..., 2, 2)."""
    a, b, c = angles[..., 0], angles[..., 1], angles[..., 2]
    ea = jnp.exp(-0.5j * a)
    ec = jnp.exp(-0.5j * c)
    cb = jnp.cos(b / 2.0).astype(CDTYPE)
    sb = jnp.sin(b / 2.0).astype(CDTYPE)
    # RZ(c) @ RY(b) @ RZ(a)
    return jnp.stack([
        jnp.stack([ec * cb * ea, -ec * sb * jnp.conj(ea)], -1),
        jnp.stack([jnp.conj(ec) * sb * ea, jnp.conj(ec) * cb * jnp.conj(ea)], -1),
    ], -2)


def expval_batched(arch: "Arch", params, X, f):
    """<Z_0> for a grid of (problem, restart) x (data point).

    params : (B, R, P)
    X      : (B, N, 2)
    f      : (B,)
    returns: (B, R, N)
    """
    B, R = params.shape[0], params.shape[1]
    N = X.shape[1]
    p = params.reshape((B, R) + arch.param_shape)          # (B,R,T,2,2,3)

    # encoding gates: one per (B, N), shared across restarts
    ex0 = _ry_b(X[:, None, :, 0])                          # (B,1,N,2,2)
    ex1 = _ry_b(X[:, None, :, 1])

    st = jnp.zeros((B, R, N, 2, 2), dtype=CDTYPE).at[..., 0, 0].set(1.0)
    cz = _CZ_MASK
    sig = arch.sigmas
    for l in range(arch.T):
        st = jnp.einsum("brnij,brnjq->brniq", ex0, st)
        st = jnp.einsum("brnij,brnpj->brnpi", ex1, st)
        a0 = _su2_b(p[:, :, l, 0, 0])[:, :, None]          # (B,R,1,2,2)
        a1 = _su2_b(p[:, :, l, 0, 1])[:, :, None]
        st = jnp.einsum("brnij,brnjq->brniq", a0, st)
        st = jnp.einsum("brnij,brnpj->brnpi", a1, st)
        st = st * cz
        if arch.kind == "ddr":
            st = st * rzz_mask_b(sig[l] * f)[:, None, None]
        elif arch.kind == "sdr":
            st = jnp.einsum("brnij,brnjq->brniq", ex0, st)
            st = jnp.einsum("brnij,brnpj->brnpi", ex1, st)
        b0 = _su2_b(p[:, :, l, 1, 0])[:, :, None]
        b1 = _su2_b(p[:, :, l, 1, 1])[:, :, None]
        st = jnp.einsum("brnij,brnjq->brniq", b0, st)
        st = jnp.einsum("brnij,brnpj->brnpi", b1, st)

    probs = jnp.abs(st) ** 2
    return jnp.sum(probs[..., 0, :], -1) - jnp.sum(probs[..., 1, :], -1)


def rzz_mask_b(phi):
    """rzz_mask on a leading batch: phi (...,) -> (..., 2, 2)."""
    m = jnp.exp(-0.5j * phi)
    p = jnp.exp(+0.5j * phi)
    return jnp.stack([jnp.stack([m, p], -1), jnp.stack([p, m], -1)], -2)


def init_params(arch: Arch, key, n: int | None = None):
    """Prior over trainable parameters: Euler angles ~ U[-pi, pi].

    The teacher parameters theta* are drawn from this same prior, so the target
    is a sample from the DDR-induced function prior.
    """
    shape = (arch.n_params,) if n is None else (n, arch.n_params)
    return jax.random.uniform(key, shape, dtype=RDTYPE, minval=-jnp.pi, maxval=jnp.pi)


# --------------------------------------------------------------------------- #
# quick self-check
# --------------------------------------------------------------------------- #

def param_table(Ts: Sequence[int] = (2,)) -> "list[dict]":
    rows = []
    for T in Ts:
        for kind in ARMS:
            rows.append(Arch(kind, T).summary())
    return rows


if __name__ == "__main__":  # pragma: no cover
    import json

    print(json.dumps(param_table((2, 3, 4)), indent=1))

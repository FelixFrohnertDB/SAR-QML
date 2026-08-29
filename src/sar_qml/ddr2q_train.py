"""Training harness for the 2-qubit DDR / SDR comparison.

The whole optimisation runs inside a single jitted `lax.scan`, vectorised over
two independent axes:

    axis 0 (B)  -- independent problems (teacher x data draw)
    axis 1 (R)  -- random restarts of the same problem

Restart selection uses the TRAINING loss only, never the test set.  An optional
scipy L-BFGS-B polish refines the selected optimum (used for the
population/approximation-error fits where precision matters).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
from scipy.optimize import minimize

from .ddr2q import Arch, RDTYPE, init_params

jax.config.update("jax_enable_x64", True)


# --------------------------------------------------------------------------- #
# batched training
# --------------------------------------------------------------------------- #


def _batched_loss_fn(arch: Arch):
    """sum_b sum_r MSE_b(params[b, r]).

    params: (B, R, P) | X: (B, N, 2) | y: (B, N) | f: (B,)
    """
    per_point = arch.expval                                     # (P,), (2,), () -> ()
    per_set = jax.vmap(per_point, in_axes=(None, 0, None))      # over N
    per_restart = jax.vmap(per_set, in_axes=(0, None, None))    # over R
    per_problem = jax.vmap(per_restart, in_axes=(0, 0, 0))      # over B

    def loss(params, X, y, f):
        pred = per_problem(params, X, f)                        # (B, R, N)
        mse = jnp.mean((pred - y[:, None, :]) ** 2, axis=-1)    # (B, R)
        return jnp.sum(mse), mse

    return loss


def _shard(*arrays_specs):
    """Shard arrays along their leading (problem) axis over all host devices."""
    from jax.sharding import Mesh, NamedSharding
    from jax.sharding import PartitionSpec as P

    nd = jax.device_count()
    mesh = Mesh(np.array(jax.devices()).reshape(nd), ("b",))
    out = []
    for arr, spec in arrays_specs:
        if arr.shape[0] % nd:
            out.append(arr)                    # not divisible: leave replicated
        else:
            out.append(jax.device_put(arr, NamedSharding(mesh, P(*spec))))
    return out


@partial(jax.jit, static_argnames=("arch", "steps", "minibatch"))
def _train_scan(arch, params0, X, y, f, steps, lr0, lr_alpha, key, minibatch):
    """Adam over `steps`, optionally on minibatches of size `minibatch`.

    The gradient uses a minibatch (cost then independent of N), but the returned
    per-restart `mse` is always the FULL training loss, so restart selection is
    unaffected by minibatch noise.
    """
    loss_fn = _batched_loss_fn(arch)
    N = X.shape[1]
    sched = optax.cosine_decay_schedule(lr0, steps, alpha=lr_alpha)
    opt = optax.adam(sched)
    state = opt.init(params0)

    def body(carry, k):
        p, s = carry
        if minibatch is not None and minibatch < N:
            idx = jax.random.choice(k, N, (minibatch,), replace=False)
            Xb, yb = X[:, idx], y[:, idx]
        else:
            Xb, yb = X, y
        (_, mse), g = jax.value_and_grad(loss_fn, has_aux=True)(p, Xb, yb, f)
        upd, s = opt.update(g, s, p)
        p = optax.apply_updates(p, upd)
        return (p, s), jnp.mean(mse)

    (p, _), trace = jax.lax.scan(body, (params0, state),
                                 jax.random.split(key, steps))
    _, mse = loss_fn(p, X, y, f)               # full-batch, for restart selection
    return p, mse, trace


def _polish(arch: Arch, p0, X, y, f, maxiter: int = 600):
    """scipy L-BFGS-B polish of a single parameter vector."""
    pred = arch.predict_fn()
    Xj, yj, fj = jnp.asarray(X), jnp.asarray(y), jnp.asarray(f, RDTYPE)

    def obj(q):
        return jnp.mean((pred(q, Xj, fj) - yj) ** 2)

    vg = jax.jit(jax.value_and_grad(obj))

    def fun(q):
        v, g = vg(jnp.asarray(q))
        return float(v), np.asarray(g, dtype=np.float64)

    res = minimize(fun, np.asarray(p0, dtype=np.float64), jac=True,
                   method="L-BFGS-B", options={"maxiter": maxiter})
    return np.asarray(res.x), float(res.fun)


@dataclass
class FitResult:
    params: np.ndarray          # (B, P) selected optimum per problem
    train_mse: np.ndarray       # (B,)
    restart_mse: np.ndarray     # (B, R) all restarts, training loss
    trace: np.ndarray           # (steps,) mean training loss
    steps: int
    n_restarts: int


def fit_batch(arch: Arch, X, y, f, *, seed=0, n_restarts=16, steps=2000,
              lr0=0.12, lr_alpha=0.02, polish=False, chunk=None,
              minibatch=64) -> FitResult:
    """Fit `arch` to B independent regression problems.

    X: (B, N, 2) | y: (B, N) | f: scalar or (B,).  Returns the best restart per
    problem, selected by the FULL training loss.

    minibatch: gradient batch size (None = full batch).  With minibatching the
    per-step cost no longer grows with N, which is what makes the large-N end of
    the learning curves affordable; restart selection still uses the full loss.
    chunk: problems per compiled call (default: all of them, sharded).
    """
    X = jnp.asarray(X, RDTYPE)
    y = jnp.asarray(y, RDTYPE)
    B = X.shape[0]
    f = jnp.broadcast_to(jnp.asarray(f, RDTYPE), (B,))
    chunk = B if chunk is None else chunk

    key = jax.random.PRNGKey(seed)
    out_p, out_train, out_restarts, traces = [], [], [], []
    for lo in range(0, B, chunk):
        hi = min(lo + chunk, B)
        k, key = jax.random.split(key)
        ks, key = jax.random.split(key)
        p0 = jax.random.uniform(k, (hi - lo, n_restarts, arch.n_params),
                                dtype=RDTYPE, minval=-jnp.pi, maxval=jnp.pi)
        p0, Xc, yc, fc = _shard((p0, ("b", None, None)),
                                (X[lo:hi], ("b", None, None)),
                                (y[lo:hi], ("b", None)),
                                (f[lo:hi], ("b",)))
        p, mse, tr = _train_scan(arch, p0, Xc, yc, fc,
                                 steps, lr0, lr_alpha, ks, minibatch)
        p, mse = np.asarray(p), np.asarray(mse)
        best = mse.argmin(axis=1)
        sel = p[np.arange(hi - lo), best]
        selm = mse[np.arange(hi - lo), best]
        if polish:
            for i in range(hi - lo):
                q, v = _polish(arch, sel[i], X[lo + i], y[lo + i], float(f[lo + i]))
                if v < selm[i]:
                    sel[i], selm[i] = q, v
        out_p.append(sel)
        out_train.append(selm)
        out_restarts.append(mse)
        traces.append(np.asarray(tr))

    return FitResult(
        params=np.concatenate(out_p),
        train_mse=np.concatenate(out_train),
        restart_mse=np.concatenate(out_restarts),
        trace=np.mean(traces, axis=0),
        steps=steps,
        n_restarts=n_restarts,
    )


# --------------------------------------------------------------------------- #
# evaluation
# --------------------------------------------------------------------------- #


def evaluate(arch: Arch, params, f, X_test, t_test, *, thresh, margin):
    """Regression + classification metrics against the *clean* teacher values.

    params: (B, P) | X_test: (B, N, 2) | t_test: (B, N) clean teacher output
    thresh: (B,) per-teacher median | margin: (B,) excluded band half-width
    Returns dict of (B,) arrays: mse, nmse, acc.
    """
    pred_fn = jax.jit(jax.vmap(jax.vmap(arch.expval, in_axes=(None, 0, None)),
                               in_axes=(0, 0, 0)))
    B = np.shape(params)[0]
    f = jnp.broadcast_to(jnp.asarray(f, RDTYPE), (B,))
    pred = np.asarray(pred_fn(jnp.asarray(params, RDTYPE),
                              jnp.asarray(X_test, RDTYPE), f))
    t = np.asarray(t_test)
    mse = np.mean((pred - t) ** 2, axis=1)
    var = np.var(t, axis=1)
    thresh = np.asarray(thresh)[:, None]
    margin = np.asarray(margin)[:, None]
    keep = np.abs(t - thresh) > margin
    acc = np.array([
        np.mean((pred[b][keep[b]] > thresh[b, 0]) == (t[b][keep[b]] > thresh[b, 0]))
        for b in range(B)
    ])
    return {"mse": mse, "nmse": mse / var, "acc": acc, "pred": pred}

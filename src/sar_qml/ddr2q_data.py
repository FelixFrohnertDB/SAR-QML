"""Teacher ensemble and dataset generation for the 2-qubit DDR experiments.

Teacher parameters theta* are drawn from the *same* prior used to initialise the
students (Euler angles ~ U[-pi, pi]), so the target function is a sample from
the DDR-induced function prior and the setup is a clean teacher-student problem.

Both label types are retained:
  * the continuous teacher expectation t(x) = <Z_0>_DDR(x, f*; theta*)  (regression)
  * a binary label from thresholding t at its own per-teacher median, with a
    margin band excluded so accuracy is not dominated by near-threshold points.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import numpy as np

from .ddr2q import Arch, init_params

# f* chosen so that BOTH effective entangling angles e_l = |wrap(sigma_l f - pi/2)|
# are far from {0, pi} (where RZZ degenerates to a product gate) and far from
# pi/2 (which is exactly the f = 0 control).  See verify_ddr2q.py / V3.
FSTAR = 5 * np.pi / 6
F_MISMATCH = FSTAR + np.pi / 2          # preregistered mismatch


@dataclass
class Teachers:
    theta: np.ndarray        # (M, P)
    fstar: float
    X_test: np.ndarray       # (M, N_test, 2)
    t_test: np.ndarray       # (M, N_test) clean
    thresh: np.ndarray       # (M,) median of t on the population grid
    margin: np.ndarray       # (M,) half-width of excluded band
    grid: np.ndarray         # (G, 2) dense population grid
    t_grid: np.ndarray       # (M, G)
    M: int

    def train_set(self, N, seed, noise=0.0):
        """Draw an independent training set of size N for every teacher.

        Evaluated in ONE compiled call over all teachers; the per-teacher Python
        loop this replaces dominated the per-cell wall-clock cost.
        """
        rng = np.random.default_rng(seed)
        X = rng.uniform(-np.pi, np.pi, (self.M, N, 2))
        t = np.asarray(_arch().predict_many(self.theta, X, self.fstar))
        y = t + noise * rng.standard_normal(t.shape) if noise > 0 else t.copy()
        return X, y, t


def _arch():
    return Arch("ddr", 2)


def population_grid(m=64):
    g = 2 * np.pi * np.arange(m) / m - np.pi
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)


def make_teachers(M=32, fstar=FSTAR, n_test=4000, seed=2026,
                  margin_frac=0.10, grid_m=64) -> Teachers:
    """Sample M teachers and their evaluation data.

    margin_frac: fraction of population points excluded around the threshold.
    """
    arch = _arch()
    theta = np.asarray(init_params(arch, jax.random.PRNGKey(seed), M))
    G = population_grid(grid_m)
    Gb = np.broadcast_to(G, (M,) + G.shape)
    t_grid = np.asarray(arch.predict_many(theta, Gb, fstar))
    thresh = np.median(t_grid, axis=1)
    # margin: half-width excluding margin_frac of the population mass
    dev = np.abs(t_grid - thresh[:, None])
    margin = np.quantile(dev, margin_frac, axis=1)

    rng = np.random.default_rng(seed + 1)
    X_test = rng.uniform(-np.pi, np.pi, (M, n_test, 2))
    t_test = np.asarray(arch.predict_many(theta, X_test, fstar))
    return Teachers(theta=theta, fstar=float(fstar), X_test=X_test, t_test=t_test,
                    thresh=thresh, margin=margin, grid=G, t_grid=t_grid, M=M)


def save(path, T: Teachers):
    np.savez_compressed(
        path, theta=T.theta, fstar=T.fstar, X_test=T.X_test, t_test=T.t_test,
        thresh=T.thresh, margin=T.margin, grid=T.grid, t_grid=T.t_grid)


def load(path) -> Teachers:
    z = np.load(path)
    return Teachers(theta=z["theta"], fstar=float(z["fstar"]), X_test=z["X_test"],
                    t_test=z["t_test"], thresh=z["thresh"], margin=z["margin"],
                    grid=z["grid"], t_grid=z["t_grid"], M=z["theta"].shape[0])

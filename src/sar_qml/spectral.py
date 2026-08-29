"""Spectral / Fourier-fingerprint analysis for the aligned-vs-standard study.

A data-reuploading PQC realises a truncated Fourier series in each parameter it
encodes (Schuld, Sweke & Meyer 2021). If we sweep ONE encoded input parameter
theta over a full period [0, 2*pi) while holding the rest fixed, the scalar
model output g(theta) is a real Fourier series whose integer-harmonic
coefficients are the model's "learned spectrum" in that direction. Comparing a
trained model's spectrum against the analytic MNUDFT teacher's spectrum along
the same scan is the mechanistic evidence for Experiment 1: an *aligned* student
should place its spectral mass where the teacher does; an x-only *standard*
student generally cannot.

This module is pure NumPy (no PennyLane / JAX dependency) so it can be imported
by notebooks and analysis scripts cheaply. Model outputs are obtained by calling
a user-supplied ``predict_fn`` (a thin closure over a trained PQCClassifier).
"""
from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import numpy as np


# ----------------------------------------------------------------------------
# 1-D Fourier fingerprint of an arbitrary scalar function of a vector input
# ----------------------------------------------------------------------------
def scan_line(
    base_point: np.ndarray,
    direction: np.ndarray,
    n_grid: int = 128,
    span: float = 2.0 * np.pi,
    mode: str = "add",
) -> Tuple[np.ndarray, np.ndarray]:
    """Build a 1-D sweep of input points.

    Parameters
    ----------
    base_point : (D,) array
        Fixed input vector; every swept coordinate starts from here.
    direction : (D,) array
        Which input coordinates to sweep and with what relative weight. For a
        single-coordinate scan use a one-hot vector; to sweep several entries in
        lockstep (e.g. all f-entries together) use ones on those entries.
    n_grid : int
        Number of grid points over one period. Use a power of two for a clean FFT.
    span : float
        Length of the swept interval (default 2*pi = one full period).
    mode : {"add", "set"}
        "add": point = base + theta * direction (harmonic sweep about base).
        "set": swept entries are SET to theta (direction acts as a 0/1 mask);
               use this to sweep an absolute encoded value over [0, span).

    Returns
    -------
    thetas : (n_grid,) array of sweep positions in [0, span).
    points : (n_grid, D) array of input vectors.
    """
    base_point = np.asarray(base_point, dtype=np.float64)
    direction = np.asarray(direction, dtype=np.float64)
    thetas = np.linspace(0.0, span, n_grid, endpoint=False)
    if mode == "set":
        mask = direction != 0.0
        points = np.tile(base_point, (n_grid, 1))
        points[:, mask] = thetas[:, None]
    else:
        points = base_point[None, :] + thetas[:, None] * direction[None, :]
    return thetas, points


def fourier_spectrum_1d(
    values: np.ndarray,
    n_freqs: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Discrete Fourier coefficients of a signal sampled uniformly over one period.

    Parameters
    ----------
    values : (G,) real array
        g(theta) sampled at theta = 2*pi*j/G, j = 0..G-1.
    n_freqs : int, optional
        If given, return only non-negative frequencies 0..n_freqs. Otherwise
        return 0..G//2.

    Returns
    -------
    freqs : (F,) int array of non-negative integer frequencies.
    coeffs : (F,) complex array; coeffs[n] is the amplitude of exp(i n theta),
        normalised so coeffs[n] = (1/G) sum_j g_j exp(-i n theta_j). The n=0 term
        is the mean (DC); |coeffs[n]| for n>=1 is the half-amplitude of that tone.
    """
    values = np.asarray(values, dtype=np.float64)
    G = values.shape[0]
    full = np.fft.fft(values) / G
    n_max = G // 2 if n_freqs is None else int(n_freqs)
    n_max = min(n_max, G // 2)
    freqs = np.arange(0, n_max + 1)
    return freqs, full[: n_max + 1]


def spectrum_of_predict(
    predict_fn: Callable[[np.ndarray], np.ndarray],
    base_point: np.ndarray,
    direction: np.ndarray,
    n_grid: int = 128,
    span: float = 2.0 * np.pi,
    mode: str = "set",
    n_freqs: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """Fourier fingerprint of a scalar model along a 1-D input scan.

    ``predict_fn`` maps a batch of input vectors (G, D) to a length-G real vector
    of scalar outputs (e.g. the raw expectation value, or a centred probability).
    Returns a dict with the sweep, the sampled output, and its spectrum.
    """
    thetas, points = scan_line(base_point, direction, n_grid=n_grid, span=span, mode=mode)
    values = np.asarray(predict_fn(points), dtype=np.float64).reshape(-1)
    freqs, coeffs = fourier_spectrum_1d(values, n_freqs=n_freqs)
    return {
        "thetas": thetas,
        "values": values,
        "freqs": freqs,
        "coeffs": coeffs,
        "power": np.abs(coeffs) ** 2,
    }


# ----------------------------------------------------------------------------
# Analytic MNUDFT teacher
# ----------------------------------------------------------------------------
def _teacher_weights(teacher: Dict) -> np.ndarray:
    return np.asarray(teacher["w_real"], dtype=np.float64) + 1j * np.asarray(
        teacher["w_imag"], dtype=np.float64
    )


def f_flat_to_pairs(f_flat: np.ndarray) -> np.ndarray:
    """Undo the interleaving [fx0, fy0, fx1, fy1, ...] -> (n_frequencies, 2)."""
    f_flat = np.asarray(f_flat, dtype=np.float64)
    return np.stack([f_flat[0::2], f_flat[1::2]], axis=1)


def mnudft_teacher_score(
    x_data: np.ndarray,
    f_data: np.ndarray,
    teacher: Dict,
) -> np.ndarray:
    """Exact teacher score s*(x; f) for a batch (real-valued, pre-threshold).

    s*(x; f) = Re( sum_k w_k * sum_d x_d * exp(-2 pi i f_k . p_d) )

    x_data : (N, n_positions); f_data : (N, 2*n_frequencies) interleaved.
    """
    x_data = np.atleast_2d(np.asarray(x_data, dtype=np.float64))
    f_data = np.atleast_2d(np.asarray(f_data, dtype=np.float64))
    w = _teacher_weights(teacher)
    positions = np.asarray(teacher["positions"], dtype=np.float64)  # (D, 2)

    scores = np.empty(x_data.shape[0], dtype=np.float64)
    for i in range(x_data.shape[0]):
        f_pairs = f_flat_to_pairs(f_data[i])                 # (K, 2)
        phase = 2.0 * np.pi * (f_pairs @ positions.T)        # (K, D)
        kernel = np.exp(-1j * phase)                         # (K, D)
        X = kernel @ x_data[i]                               # (K,)
        scores[i] = np.real(np.sum(w * X))
    return scores


def mnudft_teacher_label(x_data, f_data, teacher) -> np.ndarray:
    """Hard labels using the teacher's stored median threshold."""
    return (mnudft_teacher_score(x_data, f_data, teacher) > teacher["threshold"]).astype(np.int64)


# ----------------------------------------------------------------------------
# Experiment 2: alignment-axis helpers (perturb the FIXED query geometry)
# ----------------------------------------------------------------------------
def interpolate_f(
    f_true_flat: np.ndarray,
    f_other_flat: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Linear interpolation (1-alpha)*f_true + alpha*f_other in interleaved space.

    alpha = 0 -> the geometry the model is aligned to; alpha = 1 -> a competing
    (e.g. random) geometry. Sweeping alpha traces the accuracy vs misalignment
    curve of Experiment 2.
    """
    f_true_flat = np.asarray(f_true_flat, dtype=np.float64)
    f_other_flat = np.asarray(f_other_flat, dtype=np.float64)
    return (1.0 - alpha) * f_true_flat + alpha * f_other_flat


def rotate_query_frequencies(f_flat: np.ndarray, angle: float) -> np.ndarray:
    """Rotate every 2-D query frequency (fx_k, fy_k) by ``angle`` radians.

    A rigid rotation of the acquisition geometry: preserves the magnitude of each
    query frequency while steadily misaligning it from the fixed circuit
    structure, giving a controlled one-parameter misalignment axis for E2.
    """
    f_pairs = f_flat_to_pairs(f_flat)                        # (K, 2)
    c, s = np.cos(angle), np.sin(angle)
    R = np.array([[c, -s], [s, c]], dtype=np.float64)
    rotated = f_pairs @ R.T
    out = np.zeros_like(np.asarray(f_flat, dtype=np.float64))
    out[0::2] = rotated[:, 0]
    out[1::2] = rotated[:, 1]
    return out


def make_shared_f(f_flat: np.ndarray, n_samples: int) -> np.ndarray:
    """Tile a single interleaved f-vector across n_samples rows."""
    return np.tile(np.asarray(f_flat, dtype=np.float64)[None, :], (n_samples, 1))


def spectral_overlap(coeffs_a: np.ndarray, coeffs_b: np.ndarray) -> float:
    """Cosine similarity of two power spectra (ignoring DC), in [0, 1].

    A scalar summary of how well one model's Fourier fingerprint matches
    another's along the same scan. 1.0 = identical spectral shape.
    """
    pa = np.abs(np.asarray(coeffs_a)) ** 2
    pb = np.abs(np.asarray(coeffs_b)) ** 2
    pa = pa[1:]  # drop DC
    pb = pb[1:]
    na, nb = np.linalg.norm(pa), np.linalg.norm(pb)
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(pa, pb) / (na * nb))

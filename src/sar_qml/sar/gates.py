"""Single-qubit rotations and the fixed azimuth Fourier readout."""

import jax.numpy as jnp
import numpy as np


def ry(angle):
    """``R_Y(angle)`` as a complex 2x2 matrix."""
    c, s = jnp.cos(angle / 2), jnp.sin(angle / 2)
    return jnp.array([[c, -s], [s, c]], dtype=jnp.complex128)


def rz(angle):
    """``R_Z(angle) = diag(e^{-i angle/2}, e^{i angle/2})``."""
    return jnp.diag(jnp.array([jnp.exp(-0.5j * angle), jnp.exp(0.5j * angle)]))


def fourier_readout(q):
    """Inverse discrete Fourier transform ``exp(2 pi i j a / q) / sqrt(q)``."""
    index = np.arange(q, dtype=float)
    return jnp.asarray(np.exp(2j * np.pi * np.outer(index, index) / q) / np.sqrt(q))

"""Local omega-k acquisition geometry and the fixed query it implies.

For ``Q`` sampled range wavenumbers ``k_r`` and azimuth wavenumbers ``k_x``
the omega-k surface is ``k_y = sqrt(k_r^2 - k_x^2)``. Removing its separable
row and column means leaves the nonseparable interaction, whose leading term
around the patch centre is ``kappa_xr dk_x dk_r a~ d~`` with the mixed
derivative ``kappa_xr``. A target at reference range ``R_0`` therefore carries
the phase ``f*_geom a~ d~`` with ``f*_geom = R_0 kappa_xr dk_x dk_r``.
"""

import jax.numpy as jnp
import numpy as np

SPEED_OF_LIGHT = 299_792_458.0


def register_indices(q):
    """Register index ``0..q-1`` and its centred version ``index - (q-1)/2``."""
    index = np.arange(q, dtype=float)
    return index, index - (q - 1) / 2


def compact_interaction(q):
    """Compact bilinear generator ``a~ d~`` on a ``q x q`` patch."""
    _, centered = register_indices(q)
    return np.outer(centered, centered)


def build_geometry(
    q,
    *,
    carrier_hz,
    range_step_hz,
    kx_center,
    kx_step,
    reference_range,
    name="baseline",
):
    """Sample the local omega-k surface of one acquisition.

    Parameters are the carrier frequency, the range-frequency step (Hz), the
    azimuth-wavenumber centre (squint) and step (rad/m) and the reference
    slant range (m). Returns a dict with the derived wavenumbers, the mixed
    derivative ``kappa_xr``, ``mixed_scale = kappa_xr dk_x dk_r``, the fixed
    query ``f_geometry``, the dimensionless exact interaction ``h_exact``
    (also as ``h_exact_jax``) and the least-squares coefficient and R^2 of
    the compact generator ``a~ d~`` in ``h_exact``.
    """
    _, centered = register_indices(q)
    h_compact = compact_interaction(q)

    kr_center = 4 * np.pi * carrier_hz / SPEED_OF_LIGHT
    delta_kr = 4 * np.pi * range_step_hz / SPEED_OF_LIGHT
    kr = kr_center + delta_kr * centered
    kx = kx_center + kx_step * centered
    if not np.all(kr[None, :] ** 2 > kx[:, None] ** 2):
        raise ValueError("evanescent samples: require k_r^2 > k_x^2")

    ky = np.sqrt(kr[None, :] ** 2 - kx[:, None] ** 2)
    interaction = (
        ky
        - ky.mean(axis=1, keepdims=True)
        - ky.mean(axis=0, keepdims=True)
        + ky.mean()
    )
    kappa = kx_center * kr_center / (kr_center**2 - kx_center**2) ** 1.5
    mixed_scale = kappa * kx_step * delta_kr
    h_exact = interaction / mixed_scale
    coefficient = float(np.sum(h_exact * h_compact) / np.sum(h_compact**2))
    r_squared = float(
        1.0
        - np.sum((h_exact - coefficient * h_compact) ** 2)
        / np.sum((h_exact - h_exact.mean()) ** 2)
    )
    return {
        "name": name,
        "carrier_hz": carrier_hz,
        "range_step_hz": range_step_hz,
        "kx_center": kx_center,
        "kx_step": kx_step,
        "reference_range": reference_range,
        "kr_center": kr_center,
        "delta_kr": delta_kr,
        "kappa_xr": kappa,
        "mixed_scale": mixed_scale,
        "f_geometry": reference_range * mixed_scale,
        "h_exact": h_exact,
        "h_exact_jax": jnp.asarray(h_exact),
        "bilinear_coefficient": coefficient,
        "bilinear_r2": r_squared,
    }


def mixed_derivative_relative_error(kx_center, kr_center, kappa, epsilon=1e-3):
    """Relative error of ``kappa`` against a central finite difference."""

    def ky(kx, kr):
        return np.sqrt(kr**2 - kx**2)

    finite_mixed = (
        ky(kx_center + epsilon, kr_center + epsilon)
        - ky(kx_center + epsilon, kr_center - epsilon)
        - ky(kx_center - epsilon, kr_center + epsilon)
        + ky(kx_center - epsilon, kr_center - epsilon)
    ) / (4 * epsilon**2)
    return float(abs(finite_mixed - kappa) / abs(kappa))


def register_z_signs(n):
    """Eigenvalues ``z_i = 1 - 2 b_i`` of ``Z_i`` for every basis state.

    Returns an array of shape ``(2**n, n)``; qubit 0 is the most significant
    bit.
    """
    bits = (np.arange(2**n)[:, None] >> np.arange(n - 1, -1, -1)[None, :]) & 1
    return 1.0 - 2.0 * bits


def channel_phase(z_phases):
    """Fixed azimuth-channel phase ``exp(-i/2 sum_i phi_i z_i)`` per index."""
    z_phases = np.asarray(z_phases)
    return np.exp(-0.5j * (register_z_signs(len(z_phases)) @ z_phases))

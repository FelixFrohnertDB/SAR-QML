"""
Raw-SAR spectral classification: data generation and feature construction.

Implements the omega-k / Stolt formulation of the paper appendix:

    raw echoes  x_{ad}  sampled at  p_{ad} = (eta_a, tau_d)
    query set   f*_{ml} = ( f_eta_m , f^q_tau_ml )        [inverse Stolt]
    MNUDFT      Z_{ml}  = sum_{ad} x_{ad} exp(-2 pi i f*_{ml} . p_{ad})
    focusing    R_SAR   = F^-1_img . M_W . N_{f*}

WHAT MAKES THE QUERY SET NONUNIFORM (read this before choosing parameters)
-------------------------------------------------------------------------
Two independent effects, with very different scaling:

1. Off-grid querying. The output image grid (f_eta_m, f'_tau_l) need not
   coincide with the DFT bins of the raw block. Any query strictly between
   bins is a genuine NUDFT evaluation that no FFT of the raw block computes.
   Controlled by `oversample` and `offset`; available at ANY block size.

2. Stolt m-dependence. f^q_tau_ml depends on the azimuth index m, so the
   two-dimensional query set is non-Cartesian. Measured in raw
   range-frequency bins this effect scales as

       stolt_bins  ~  n_rg * f0 * sin^2(theta_max) / (2 * fs),

   i.e. PROPORTIONAL TO n_rg. For a spaceborne C-band block with n_rg ~ 4096
   it is ~4 bins and Stolt interpolation is mandatory. For n_rg <~ 64 it is
   below 0.01 bins for any geometry that also satisfies azimuth Nyquist: a
   short fast-time record simply cannot resolve that frequency difference.

So at quantum-feasible block sizes the query set is non-Cartesian only in
sense (1), not (2). `diagnostics()` reports both and says so. Do not claim
the Stolt nonuniformity is doing work at n_rg = 8; it is not.

Sign / reference conventions
----------------------------
* eta_a and tau_d are measured relative to the centre of the acquired block
  (aperture centre, reference delay 2*R0/c). The dropped absolute reference
  delay contributes a phase depending only on f_tau, which is exactly what the
  omega-k reference function H compensates; it is absorbed into W.
* W is built as the phase conjugate of the response of a unit scatterer at the
  scene reference point (eta=0, R=R0). This is the matched filter for the
  assumed geometry and avoids hand-deriving H.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

C_LIGHT = 299_792_458.0


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------


@dataclass(frozen=True)
class Geometry:
    """Monostatic broadside stripmap geometry with the hyperbolic range model."""

    f0: float  # carrier frequency [Hz]
    fs: float  # range sampling rate ~ bandwidth [Hz]
    prf: float  # pulse repetition frequency [Hz]
    v_r: float  # effective platform velocity [m/s]
    r0: float  # reference slant range [m]
    n_az: int  # azimuth samples in the raw block
    n_rg: int  # range samples in the raw block

    @property
    def wavelength(self) -> float:
        return C_LIGHT / self.f0

    @property
    def eta(self) -> np.ndarray:
        """Azimuth slow time relative to aperture centre [s]."""
        return (np.arange(self.n_az) - self.n_az // 2) / self.prf

    @property
    def tau(self) -> np.ndarray:
        """Fast time relative to the reference delay 2*r0/c [s]."""
        return (np.arange(self.n_rg) - self.n_rg // 2) / self.fs

    @property
    def d_f_eta(self) -> float:
        """Azimuth-frequency resolution of the raw block (DFT bin width) [Hz]."""
        return self.prf / self.n_az

    @property
    def d_f_tau(self) -> float:
        """Range-frequency resolution of the raw block (DFT bin width) [Hz]."""
        return self.fs / self.n_rg

    @property
    def synthetic_aperture(self) -> float:
        return self.n_az * self.v_r / self.prf

    @property
    def integration_angle(self) -> float:
        """Aspect angle at the aperture edge relative to broadside [rad]."""
        return self.synthetic_aperture / (2.0 * self.r0)

    @property
    def doppler_bandwidth(self) -> float:
        """Doppler bandwidth actually collected [Hz]."""
        return 4.0 * self.v_r * np.sin(self.integration_angle) / self.wavelength

    @property
    def azimuth_oversampling(self) -> float:
        """prf / doppler_bandwidth. Should be ~1: below 1 the azimuth signal
        aliases, far above 1 most of the azimuth frequency grid carries no
        signal and azimuth compression is doing nothing."""
        return self.prf / self.doppler_bandwidth

    @property
    def range_resolution(self) -> float:
        return C_LIGHT / (2.0 * self.fs)

    @property
    def rcm_range_cells(self) -> float:
        """Range cell migration over the aperture, in range resolution cells.

        This is what delocalises a point scatterer in the RAW coordinates and
        is the property the classification task relies on. It is NOT the same
        quantity as the Stolt frequency shift; see the module docstring.
        """
        th = self.integration_angle
        return self.r0 * (1.0 / np.cos(th) - 1.0) / self.range_resolution

    @property
    def positions(self) -> np.ndarray:
        """p_{ad} = (eta_a, tau_d), shape (n_az * n_rg, 2)."""
        e, t = np.meshgrid(self.eta, self.tau, indexing="ij")
        return np.stack([e.ravel(), t.ravel()], axis=-1)

    def range_history(self, eta, eta_c: float, r_c: float) -> np.ndarray:
        return np.sqrt(r_c**2 + (self.v_r * (eta - eta_c)) ** 2)

    def report(self) -> str:
        return (
            f"  f0={self.f0/1e6:.1f} MHz  lam={self.wavelength:.3f} m  "
            f"fs={self.fs/1e6:.1f} MHz  PRF={self.prf:.2f} Hz\n"
            f"  v_r={self.v_r:.1f} m/s  R0={self.r0/1e3:.3f} km  "
            f"block={self.n_az}x{self.n_rg}  L_sa={self.synthetic_aperture:.1f} m\n"
            f"  integration angle={np.degrees(self.integration_angle):.2f} deg  "
            f"azimuth oversampling={self.azimuth_oversampling:.2f}  "
            f"RCM={self.rcm_range_cells:.2f} range cells"
        )


def design_geometry(
    n_az: int,
    n_rg: int,
    wavelength: float = 5.45,
    fs: float = 70e6,
    v_r: float = 100.0,
    r0: float = 3.0e3,
) -> Geometry:
    """Choose the PRF so azimuth sampling is Nyquist-consistent.

    Requiring prf = doppler_bandwidth with L_sa = n_az * v_r / prf and
    theta_max = L_sa / (2 r0) gives, in the small-angle limit,

        prf = v_r * sqrt( 2 n_az / (r0 * lambda) ),

    for which the range cell migration over the block works out to

        RCM  =  n_az * lambda * fs / (4 c)   range cells,

    so a long wavelength and a wide bandwidth are what keep a point scatterer
    delocalised in the raw coordinates at small n_az. Low-frequency UWB SAR
    (the classic omega-k motivating case) is the only regime where this holds
    on a small block.

    Without this the aperture and the PRF are set independently and the
    azimuth frequency grid mostly covers empty band, which silently makes the
    azimuth half of the query set meaningless.
    """
    prf = v_r * np.sqrt(2.0 * n_az / (r0 * wavelength))
    return Geometry(
        f0=C_LIGHT / wavelength,
        fs=fs,
        prf=prf,
        v_r=v_r,
        r0=r0,
        n_az=n_az,
        n_rg=n_rg,
    )


# ----------------------------------------------------------------------------
# query set
# ----------------------------------------------------------------------------


@dataclass
class QuerySet:
    """Geometry-determined query frequencies f*_{ml} and their diagnostics."""

    geom: Geometry
    f_eta_out: np.ndarray = field(repr=False)  # (M,) output azimuth grid
    f_tau_out: np.ndarray = field(repr=False)  # (L,) output range grid (f'_tau)
    f_q_tau: np.ndarray = field(repr=False)  # (M, L) inverse-Stolt range freqs
    mask: np.ndarray = field(repr=False)  # (M, L) inside acquired band
    f_star: np.ndarray = field(repr=False)  # (K, 2) retained queries
    idx: np.ndarray = field(repr=False)  # (K, 2) output-grid indices
    stolt_bins: float = 0.0  # azimuth-dependence of f^q, in raw range bins
    offgrid_bins: float = 0.0  # median distance of queries from raw DFT bins
    retained_fraction: float = 0.0


def inverse_stolt(geom: Geometry, f_eta, f_tau_out) -> np.ndarray:
    """f^q_tau = sqrt( (f0 + f'_tau)^2 + (c f_eta / (2 v_r))^2 ) - f0."""
    lateral = C_LIGHT * np.asarray(f_eta)[:, None] / (2.0 * geom.v_r)
    return np.sqrt((geom.f0 + np.asarray(f_tau_out)[None, :]) ** 2 + lateral**2) - geom.f0


def build_query_set(
    geom: Geometry,
    oversample: float = 1.0,
    offset: tuple = (0.5, 0.5),
    n_az_out: int | None = None,
    n_rg_out: int | None = None,
) -> QuerySet:
    """Build the retained, geometry-prescribed query set.

    Parameters
    ----------
    oversample : output grid spacing relative to the raw DFT bin width. The
        output image grid of an omega-k processor is a processing choice and
        need not match the raw block's DFT bins.
    offset : fractional shift of the output grid in (azimuth, range) DFT bins.
        (0.5, 0.5) puts every query midway between raw DFT bins, which is the
        maximally off-grid choice and the strongest available form of query
        nonuniformity at small block sizes. (0.0, 0.0) reproduces the raw DFT
        bins, in which case the MNUDFT collapses to an ordinary FFT.
    """
    n_az_out = geom.n_az if n_az_out is None else n_az_out
    n_rg_out = geom.n_rg if n_rg_out is None else n_rg_out

    f_eta_out = (
        np.arange(n_az_out) - n_az_out // 2 + offset[0]
    ) * geom.d_f_eta / oversample
    f_tau_out = (
        np.arange(n_rg_out) - n_rg_out // 2 + offset[1]
    ) * geom.d_f_tau / oversample

    f_q = inverse_stolt(geom, f_eta_out, f_tau_out)

    lo, hi = geom.tau.size and -geom.fs / 2, geom.fs / 2
    mask = (f_q >= lo) & (f_q <= hi)

    m_idx, l_idx = np.nonzero(mask)
    f_star = np.stack([f_eta_out[m_idx], f_q[m_idx, l_idx]], axis=-1)
    idx = np.stack([m_idx, l_idx], axis=-1)

    stolt = (f_q.max(axis=0) - f_q.min(axis=0)).max() / geom.d_f_tau

    # distance of each query from the nearest raw DFT bin, in bins
    def _frac(v, dv):
        r = np.abs(v / dv - np.round(v / dv))
        return r

    off = np.hypot(
        _frac(f_star[:, 0], geom.d_f_eta), _frac(f_star[:, 1], geom.d_f_tau)
    )

    return QuerySet(
        geom=geom,
        f_eta_out=f_eta_out,
        f_tau_out=f_tau_out,
        f_q_tau=f_q,
        mask=mask,
        f_star=f_star,
        idx=idx,
        stolt_bins=float(stolt),
        offgrid_bins=float(np.median(off)),
        retained_fraction=float(mask.mean()),
    )


def nondimensionalise(f_star: np.ndarray, positions: np.ndarray):
    """Diagonal rescaling f -> A f, p -> A^-T p; every Fourier phase preserved.

    A is fixed by requiring the two coordinates to contribute comparably and
    the largest phase 2 pi f.p to be of order pi, so the encoded rotation
    angles fill the fundamental domain without wrapping. Returns
    (f_bar, p_bar, A_diag). This is a stated convention, not a tuned knob.
    """
    fmax = np.abs(f_star).max(axis=0)
    pmax = np.abs(positions).max(axis=0)
    a_diag = np.where((fmax > 0) & (pmax > 0), np.sqrt(fmax / np.where(pmax > 0, pmax, 1.0)), 1.0)
    f_bar, p_bar = f_star / a_diag, positions * a_diag
    peak = 2 * np.pi * np.abs(f_bar @ p_bar.T).max()
    if peak > 0:
        g = np.sqrt(np.pi / peak)
        f_bar, p_bar = f_bar * g, p_bar * g
    return f_bar, p_bar, a_diag




# ----------------------------------------------------------------------------
# PQC-facing coordinates
# ----------------------------------------------------------------------------


def lattice_nondimensionalise(geom: Geometry, s: float = 1.0):
    """Rescaling that puts the sampling positions on an INTEGER lattice.

        A = diag(prf, fs) * s ,     p_bar = A p ,    f_bar = A^-T f

    Every Fourier phase is preserved, 2 pi f.p = 2 pi f_bar.p_bar, and because
    eta_a and tau_d are uniform grids with spacings 1/prf and 1/fs,

        p_bar_{ad} = s * ( a - n_az//2 , d - n_rg//2 )   -- integers times s.

    Why this matters. Omega^(f) for commuting Z-type generators with scales
    sigma_l is { sum_l m_l sigma_l : m_l in {-2,0,2} }. Choosing

        sigma_l = 2^(l-1) * s / 2 ,    l = 1..T

    makes that set EXACTLY s * {-(2^T - 1), ..., 2^T - 1}, since digits
    {-1,0,1} with binary weights represent every integer in range. So with
    T = ceil(log2(n)) query layers per coordinate the task modes lie in
    Omega^(f) exactly -- the realizability bound is met with equality and the
    2^-T approximation error vanishes.

    The generic `nondimensionalise()` fills the fundamental domain but returns
    arbitrary reals, which throws that exactness away. Prefer this one whenever
    the positions are a uniform grid.
    """
    a_diag = np.array([geom.prf, geom.fs]) * s
    p_bar = geom.positions * a_diag
    return p_bar, a_diag


def dyadic_scales(n_points: int, s: float = 1.0) -> np.ndarray:
    """Generator scales sigma_l reaching every integer offset on a grid of n."""
    t = int(np.ceil(np.log2(max(n_points, 2))))
    return np.array([2.0 ** (l - 1) * s / 2.0 for l in range(1, t + 1)])


def pqc_spec(geom: Geometry, qs: QuerySet, f_set: np.ndarray, s: float = 1.0) -> dict:
    """Register and layer budget for uploading `f_set`, plus an exactness check.

    Returns the number of query layers T per coordinate, the number of qubit
    groups (one per (query, coordinate) pair, since the task modes are the
    lifted vectors e_k (x) p_d), the dimensionless query angles, and the
    residual between the required position lattice and Omega^(f).
    """
    p_bar, a_diag = lattice_nondimensionalise(geom, s)
    f_bar = f_set / a_diag

    spec = {"p_bar": p_bar, "f_bar": f_bar, "a_diag": a_diag, "s": s}
    for e, (name, n) in enumerate((("azimuth", geom.n_az), ("range", geom.n_rg))):
        sig = dyadic_scales(n, s)
        reach = (2 ** len(sig) - 1) * s
        need = np.abs(p_bar[:, e]).max()
        # every accessible offset, from digits {-1,0,1} on binary weights
        spec[name] = {
            "T": len(sig),
            "sigma": sig,
            "lattice_reach": reach,
            "max_position": need,
            "exact": bool(
                need <= reach + 1e-12
                and np.allclose(p_bar[:, e] / s, np.round(p_bar[:, e] / s))
            ),
        }
    k = f_set.shape[0]
    spec["n_query_groups"] = 2 * k
    spec["n_query_layers"] = max(spec["azimuth"]["T"], spec["range"]["T"])
    spec["max_query_angle"] = float(2 * np.pi * np.abs(f_bar).max())
    return spec


def encode_angles(z: np.ndarray, r_ref: float | None = None) -> np.ndarray:
    """Magnitude/phase upload angles -- the representation the DDR circuit sees.

    For a quadratic decision the target coefficient functions are
    x_d x*_d' = r_d r_d' exp(i(phi_d - phi_d')). Under magnitude/phase upload
    the phase factor is EXACTLY a unit Fourier mode, so a single R_Z layer
    already covers Omega^(x) = {-1,0,1} for it and only the r_d r_d' envelope
    has to be learned. Under a real/imaginary split every factor is polynomial
    and must be approximated trigonometrically, so approximation error is paid
    on the part that could have been free.

    Returns [theta_magnitude, theta_phase], both in [-pi, pi].
    """
    r = np.abs(z)
    if r_ref is None:
        r_ref = np.quantile(r, 0.99)
    theta_r = np.pi * np.clip(r / max(r_ref, 1e-30), 0.0, 1.0)
    return np.concatenate([theta_r, np.angle(z)], axis=-1)


# ----------------------------------------------------------------------------
# simulation / transforms
# ----------------------------------------------------------------------------


def simulate_raw(geom: Geometry, eta_c, r_c, amp) -> np.ndarray:
    """Raw complex baseband echoes for point scatterers, shape (n_az, n_rg).

    Built in the range-frequency domain, where a bandlimited impulse pulse is
    a pure phase, then transformed to fast time:

        S(eta, f_tau) = sum_i a_i exp(-2 pi i (f0 + f_tau) dtau_i(eta))
        x(eta, tau)   = (1/n_rg) sum_l S(eta, f_tau_l) exp(+2 pi i f_tau_l tau)

    with dtau_i(eta) = 2 R_i(eta)/c - 2 R0/c. Exact for the hyperbolic range
    model: no POSP approximation and no range-frequency aliasing.

    The azimuth window is rectangular over the whole acquired block, so every
    scatterer contributes to every azimuth sample -- the response is fully
    delocalised in the raw coordinates, which is the point.
    """
    eta_c, r_c, amp = map(np.atleast_1d, (eta_c, r_c, amp))
    f_tau = (np.arange(geom.n_rg) - geom.n_rg // 2) * geom.fs / geom.n_rg
    tau_ref = 2.0 * geom.r0 / C_LIGHT

    r_of_eta = np.stack(
        [geom.range_history(geom.eta, ec, rc) for ec, rc in zip(eta_c, r_c)], axis=0
    )
    d_tau = 2.0 * r_of_eta / C_LIGHT - tau_ref  # (n_scat, n_az)

    phase = -2j * np.pi * (geom.f0 + f_tau[None, None, :]) * d_tau[:, :, None]
    s_f = np.einsum("i,iad->ad", amp.astype(complex), np.exp(phase))

    kernel = np.exp(2j * np.pi * np.outer(f_tau, geom.tau))
    return (s_f @ kernel) / geom.n_rg


def mnudft(x: np.ndarray, f_star: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Z_k = sum_{ad} x_{ad} exp(-2 pi i f*_k . p_{ad}). Batched over leading axes."""
    kernel = np.exp(-2j * np.pi * (f_star @ positions.T))
    return x.reshape(*x.shape[:-2], -1) @ kernel.T


def reference_function(geom: Geometry, qs: QuerySet) -> np.ndarray:
    """Phase-only matched filter W_k from the reference-target response."""
    x_ref = simulate_raw(geom, 0.0, geom.r0, 1.0 + 0j)
    z_ref = mnudft(x_ref, qs.f_star, geom.positions)
    return np.conj(z_ref) / np.where(np.abs(z_ref) > 0, np.abs(z_ref), 1.0)


def image_coords(qs: QuerySet):
    """Image-domain (eta_u, tau'_v) grids dual to the output frequency grids."""
    m, l = qs.f_eta_out.size, qs.f_tau_out.size
    d_fe = qs.f_eta_out[1] - qs.f_eta_out[0] if m > 1 else 1.0
    d_ft = qs.f_tau_out[1] - qs.f_tau_out[0] if l > 1 else 1.0
    eta_u = (np.arange(m) - m // 2) / (m * d_fe)
    tau_v = (np.arange(l) - l // 2) / (l * d_ft)
    return eta_u, tau_v


def focus(x: np.ndarray, geom: Geometry, qs: QuerySet, w: np.ndarray) -> np.ndarray:
    """Focused image via the direct-MNUDFT route: F^-1_img . M_W . N_{f*}.

    Retained queries are scattered onto the uniform (f_eta, f'_tau) output grid
    and inverse-transformed. Non-retained points are zero-filled, the usual
    omega-k valid-region treatment.
    """
    z = mnudft(x, qs.f_star, geom.positions) * w
    m, l = qs.f_eta_out.size, qs.f_tau_out.size
    spec = np.zeros((*z.shape[:-1], m, l), dtype=complex)
    spec[..., qs.idx[:, 0], qs.idx[:, 1]] = z

    eta_u, tau_v = image_coords(qs)
    az = np.exp(2j * np.pi * np.outer(qs.f_eta_out, eta_u))
    rg = np.exp(2j * np.pi * np.outer(qs.f_tau_out, tau_v))
    return np.einsum("...ml,mu,lv->...uv", spec, az, rg)


# ----------------------------------------------------------------------------
# diagnostics
# ----------------------------------------------------------------------------


def cartesian_control(geom: Geometry, qs: QuerySet) -> np.ndarray:
    """Query set with the Stolt m-dependence removed: f^q_tau -> f'_tau."""
    return np.stack(
        [qs.f_eta_out[qs.idx[:, 0]], qs.f_tau_out[qs.idx[:, 1]]], axis=-1
    )


def dft_grid_control(geom: Geometry, qs: QuerySet) -> np.ndarray:
    """Query set snapped to the raw block's DFT bins: removes off-grid-ness."""
    f = qs.f_star.copy()
    f[:, 0] = np.round(f[:, 0] / geom.d_f_eta) * geom.d_f_eta
    f[:, 1] = np.round(f[:, 1] / geom.d_f_tau) * geom.d_f_tau
    return f


def focusing_error(geom: Geometry, qs: QuerySet, f_alt: np.ndarray) -> float:
    """Relative image error from focusing with an alternative query set.

    Empirical answer to "does this piece of the geometry actually matter?":
    focus a reference target with f* and with f_alt, using the SAME matched
    filter W, and compare. Near zero means the two query sets are
    interchangeable on this block and the distinction is not doing work.
    """
    x = simulate_raw(geom, 0.0, geom.r0, 1.0 + 0j)
    w = reference_function(geom, qs)
    ref = focus(x, geom, qs, w)

    qs_alt = QuerySet(
        geom=geom,
        f_eta_out=qs.f_eta_out,
        f_tau_out=qs.f_tau_out,
        f_q_tau=qs.f_q_tau,
        mask=qs.mask,
        f_star=f_alt,
        idx=qs.idx,
    )
    alt = focus(x, geom, qs_alt, w)
    return float(np.linalg.norm(alt - ref) / np.linalg.norm(ref))


def diagnostics(geom: Geometry, qs: QuerySet) -> str:
    """The numbers that decide whether the experiment is about SAR."""
    e_stolt = focusing_error(geom, qs, cartesian_control(geom, qs))
    e_grid = focusing_error(geom, qs, dft_grid_control(geom, qs))

    out = [
        "GEOMETRY",
        geom.report(),
        "",
        "QUERY SET",
        f"  retained {qs.f_star.shape[0]} of {qs.f_eta_out.size * qs.f_tau_out.size}"
        f" ({100*qs.retained_fraction:.0f}%)",
        f"  off-grid distance (median)      : {qs.offgrid_bins:.3f} raw DFT bins"
        f"   -> focusing error {e_grid:.3e}",
        f"  Stolt azimuth-dependence of f^q : {qs.stolt_bins:.4f} raw DFT bins"
        f"   -> focusing error {e_stolt:.3e}",
        "",
        "VERDICT",
    ]
    if qs.offgrid_bins > 0.1 and e_grid > 1e-2:
        out.append(
            "  Queries are genuinely off the raw DFT grid: the MNUDFT is not an FFT."
        )
    else:
        out.append(
            "  WARNING: queries sit on the raw DFT bins. The MNUDFT reduces to an "
            "ordinary FFT and nothing is nonuniform. Set offset=(0.5, 0.5)."
        )
    if qs.stolt_bins > 1.0 and e_stolt > 1e-2:
        out.append("  Stolt remapping is resolvable on this block and is doing work.")
    else:
        out.append(
            f"  Stolt remapping is NOT resolvable on this block ({qs.stolt_bins:.4f} "
            "bins). It scales as n_rg, so this is a block-length limit, not a\n"
            "  parameter choice. Do not claim the non-Cartesian structure is active "
            "here; report the off-grid nonuniformity instead."
        )
    if not 0.5 < geom.azimuth_oversampling < 2.0:
        out.append(
            f"  WARNING: azimuth oversampling {geom.azimuth_oversampling:.2f} is far "
            "from 1; the azimuth frequency grid does not match the collected "
            "Doppler band. Use design_geometry()."
        )
    if geom.rcm_range_cells < 1.0:
        out.append(
            f"  WARNING: RCM {geom.rcm_range_cells:.2f} < 1 range cell; scatterers "
            "are not delocalised in range and the task is too easy from raw data."
        )
    return "\n".join(out)


# ----------------------------------------------------------------------------
# the classification task
# ----------------------------------------------------------------------------


@dataclass
class Dataset:
    """Real-valued feature blocks for one draw of the two-scatterer task.

    Complex quantities are split into real and imaginary parts, the bijective
    coordinate change of the paper appendix.
    """

    y: np.ndarray
    raw: np.ndarray  # 2 * n_az * n_rg
    query_full: np.ndarray  # 2 * K, all retained queries
    query_sub: np.ndarray  # 2 * K_sub, register-feasible subset
    query_perm: np.ndarray  # 2 * K_sub, permuted-f* null (arm N1)
    query_rand: np.ndarray  # 2 * K_sub, random-f null (arm N2)
    query_grid: np.ndarray  # 2 * K_sub, DFT-grid null (off-grid removed)
    raw_lin: np.ndarray  # raw @ A^T, a random invertible linear map (control)
    raw_ang: np.ndarray  # magnitude/phase upload angles -- what the DDR sees
    image_int: np.ndarray  # |i_uv|^2, fully processed reference
    meta: dict = field(default_factory=dict)

    def blocks(self) -> dict:
        return {
            k: getattr(self, k)
            for k in (
                "raw",
                "query_full",
                "query_sub",
                "query_perm",
                "query_rand",
                "query_grid",
                "raw_lin",
                "raw_ang",
                "image_int",
            )
        }


def _as_real(z: np.ndarray) -> np.ndarray:
    return np.concatenate([z.real, z.imag], axis=-1)


def select_subset(qs: QuerySet, n_az_q: int = 2, n_rg_q: int = 2) -> np.ndarray:
    """Spread-out subset of retained queries, size <= n_az_q * n_rg_q.

    The register bounds K: with one f coordinate per qubit and T query layers,
    2K components need 2K/T qubits. At least two distinct m and two distinct l
    must survive, or the inverse-Stolt m-dependence drops out entirely.
    """
    m_all, l_all = np.unique(qs.idx[:, 0]), np.unique(qs.idx[:, 1])
    m_pick = m_all[np.linspace(0, len(m_all) - 1, n_az_q).round().astype(int)]
    l_pick = l_all[np.linspace(0, len(l_all) - 1, n_rg_q).round().astype(int)]

    sel = []
    for m in m_pick:
        cand = np.nonzero(qs.idx[:, 0] == m)[0]
        if not cand.size:
            continue
        for l in l_pick:
            sel.append(cand[np.argmin(np.abs(qs.idx[cand, 1] - l))])
    sel = np.array(sorted(set(sel)))
    if n_az_q > 1 and n_rg_q > 1 and (
        len(np.unique(qs.idx[sel, 0])) < 2 or len(np.unique(qs.idx[sel, 1])) < 2
    ):
        raise ValueError("query subset is Cartesian-degenerate")
    return sel


def build_arms(
    geom: Geometry,
    qs: QuerySet,
    n_az_q: int = 2,
    n_rg_q: int = 2,
    seed: int = 12345,
) -> dict:
    """Fixed query sets for the aligned arm and its nulls.

    These are part of the MODEL specification, not of the data, so they are
    built once from a dedicated seed and shared by every split. Mirrors the
    circuit arms of the paper:

        f_sub   arm A   geometry-prescribed queries, register-feasible subset
        f_perm  arm N1  components of f* permuted across (m, l) blocks:
                        identical multiset of encoded frequencies, geometric
                        correspondence destroyed
        f_rand  arm N2  frequencies drawn uniformly over the same support:
                        controls for the mere presence of a phase pattern
        f_grid  ---     f* snapped to the raw DFT bins: removes the off-grid
                        nonuniformity while keeping the geometry
    """
    rng = np.random.default_rng(seed)
    sel = select_subset(qs, n_az_q, n_rg_q)
    f_sub = qs.f_star[sel]
    return dict(
        sel=sel,
        f_sub=f_sub,
        f_perm=f_sub.ravel()[rng.permutation(f_sub.size)].reshape(f_sub.shape),
        f_rand=np.stack(
            [
                rng.uniform(qs.f_star[:, 0].min(), qs.f_star[:, 0].max(), len(sel)),
                rng.uniform(qs.f_star[:, 1].min(), qs.f_star[:, 1].max(), len(sel)),
            ],
            axis=-1,
        ),
        f_grid=dft_grid_control(geom, qs)[sel],
        # Control for the representation claim. A random invertible linear map
        # of the raw block preserves ALL information and, for an unregularised
        # quadratic model, all performance. Any gap between raw_lin and
        # query_full therefore cannot be information or dimensionality; it can
        # only be basis dependence of the norm penalty.
        a_ctrl=_random_invertible(2 * geom.n_az * geom.n_rg, rng),
    )


def _random_invertible(d: int, rng) -> np.ndarray:
    a = rng.standard_normal((d, d)) / np.sqrt(d)
    return a


def make_dataset(
    geom: Geometry,
    qs: QuerySet,
    n_samples: int,
    sep_cells: float = 2.0,
    n_az_q: int = 2,
    n_rg_q: int = 2,
    snr_db: float | None = 20.0,
    jitter_cells: float = 0.0,
    scene: str = "full",
    clutter_db: float = -6.0,
    arms: dict | None = None,
    rng: np.random.Generator | None = None,
) -> Dataset:
    """Two point scatterers, random complex amplitudes, label = the brighter one.

    Scene: two scatterers at the same slant range, separated in azimuth by
    `sep_cells` image cells. Amplitudes are drawn per sample; the label is

        y = sign(|alpha_1|^2 - |alpha_2|^2)

    which is exactly G(MNUDFT(.)) with G a quadratic polynomial, so the exact
    (not Stone-Weierstrass) branch of the nested-Fourier proposition applies.

    The two raw responses overlap completely across the aperture, so nothing
    local in raw coordinates separates them. In query coordinates the two
    unknown complex amplitudes are recoverable from as few as two consistent
    queries, so the SAME quadratic decision rule needs a 4-dimensional rather
    than a 2*n_az*n_rg-dimensional feature space. That gap is the claim.

    Two design constraints, both learned the hard way:

    * The pair is ENERGY BALANCED (|a1|^2 + |a2|^2 fixed). Otherwise the label
      is readable from the raw envelope magnitudes alone and a linear model on
      magnitude/phase upload angles solves it without any focusing.
    * `scene="full"` fills every remaining azimuth cell with independent
      clutter, so the scene dimension exceeds the query count. With only two
      scatterers the data lies on a 2-complex-dimensional subspace, any generic
      set of >=2 projections determines it, and query alignment provably
      cannot matter -- permuted and random query sets score as well as the
      geometry-prescribed one.

    `jitter_cells` > 0 randomises the pair's position per sample, which
    forecloses the objection that a classifier exploited a single fixed pair.

    `arms` MUST be shared between the training and test datasets: the null
    query sets are part of the model specification, not of the sample, so
    redrawing them per split silently changes the feature map between train
    and test and produces meaningless (even below-chance) scores.
    """
    rng = np.random.default_rng() if rng is None else rng
    eta_u, _ = image_coords(qs)
    d_eta = eta_u[1] - eta_u[0]
    m_cells = qs.f_eta_out.size

    # Energy-balanced pair: |a1|^2 + |a2|^2 is CONSTANT and only the split
    # varies. Without this the label leaks into the raw envelope magnitudes and
    # a linear model on |x_ad| solves the task with no focusing at all.
    t = rng.uniform(0.0, 0.5 * np.pi, n_samples)
    a1 = np.cos(t) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_samples))
    a2 = np.sin(t) * np.exp(1j * rng.uniform(0, 2 * np.pi, n_samples))
    y = np.sign(np.abs(a1) ** 2 - np.abs(a2) ** 2).astype(int)

    centre = rng.uniform(-jitter_cells, jitter_cells, n_samples) * d_eta
    eta1 = centre - 0.5 * sep_cells * d_eta
    eta2 = centre + 0.5 * sep_cells * d_eta

    # Scene dimension must EXCEED the query count, or any generic set of
    # projections determines the whole scene and alignment cannot matter.
    # `scene="full"` occupies every azimuth cell with an independent random
    # amplitude, so a query set that does not isolate the two named cells
    # mixes clutter into the decision.
    u1 = int(round(m_cells // 2 - 0.5 * sep_cells))
    u2 = int(round(m_cells // 2 + 0.5 * sep_cells))
    others = [u for u in range(m_cells) if u not in (u1, u2)]
    cl = 10.0 ** (clutter_db / 20.0)

    x = np.empty((n_samples, geom.n_az, geom.n_rg), dtype=complex)
    for i in range(n_samples):
        pos, amp = [eta1[i], eta2[i]], [a1[i], a2[i]]
        if scene == "full" and others:
            pos += [eta_u[u] + centre[i] for u in others]
            amp += list(
                cl
                * (
                    rng.standard_normal(len(others))
                    + 1j * rng.standard_normal(len(others))
                )
                / np.sqrt(2)
            )
        x[i] = simulate_raw(geom, pos, [geom.r0] * len(pos), amp)

    if snr_db is not None:
        sigma = np.sqrt((np.abs(x) ** 2).mean()) * 10 ** (-snr_db / 20.0)
        x = x + (sigma / np.sqrt(2)) * (
            rng.standard_normal(x.shape) + 1j * rng.standard_normal(x.shape)
        )

    if arms is None:
        arms = build_arms(geom, qs, n_az_q=n_az_q, n_rg_q=n_rg_q)
    sel = arms["sel"]
    f_sub, f_perm = arms["f_sub"], arms["f_perm"]
    f_rand, f_grid = arms["f_rand"], arms["f_grid"]

    p = geom.positions
    w = reference_function(geom, qs)
    raw_real = _as_real(x.reshape(n_samples, -1))

    return Dataset(
        y=y,
        raw=raw_real,
        query_full=_as_real(mnudft(x, qs.f_star, p)),
        query_sub=_as_real(mnudft(x, f_sub, p)),
        query_perm=_as_real(mnudft(x, f_perm, p)),
        query_rand=_as_real(mnudft(x, f_rand, p)),
        query_grid=_as_real(mnudft(x, f_grid, p)),
        raw_lin=raw_real @ arms["a_ctrl"].T,
        raw_ang=encode_angles(x.reshape(n_samples, -1), r_ref=arms.get("r_ref")),
        image_int=(np.abs(focus(x, geom, qs, w)) ** 2).reshape(n_samples, -1),
        meta=dict(
            sel=sel, f_sub=f_sub, f_perm=f_perm, f_rand=f_rand, f_grid=f_grid,
            eta1=eta1, eta2=eta2, a1=a1, a2=a2, sep_cells=sep_cells, snr_db=snr_db,
        ),
    )


def label_consistency(ds: Dataset, geom: Geometry, qs: QuerySet) -> float:
    """Fraction of samples whose FOCUSED IMAGE agrees with the label.

    Verifies that the amplitude-defined label is the label a downstream
    classifier would assign on the focused image, i.e. that focusing actually
    resolves the pair on this grid. If this is not close to 1, increase
    `sep_cells` or the block size -- otherwise the task is not the task.
    """
    m, l = qs.f_eta_out.size, qs.f_tau_out.size
    img = ds.image_int.reshape(-1, m, l)
    sep = ds.meta["sep_cells"]
    u1, u2 = int(round(m // 2 - 0.5 * sep)), int(round(m // 2 + 0.5 * sep))
    v = l // 2
    pred = np.sign(img[:, u1, v] - img[:, u2, v]).astype(int)
    return float((pred == ds.y).mean())

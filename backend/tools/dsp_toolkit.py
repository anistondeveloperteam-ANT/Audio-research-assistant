"""
dsp_toolkit.py  --  Batch 12 (Phase 2)

Pre-built, verified, dimension-checked DSP functions for the LLM to
call from inside the sandboxed code executor. The whole point of
this module is to NOT make the LLM re-derive the math.

CONVENTIONS  (used uniformly across every function)
---------------------------------------------------
  - `rx_signals` has shape (n_sensors, n_snapshots).
        Sensors along rows. Time/snapshots along columns.
  - Sample covariance:
        R = (1.0 / n_snapshots) * X @ X.conj().T     # shape (n_sensors, n_sensors)
  - Steering vector for a uniform linear array (ULA) with element
    spacing d, wavelength lam, look direction theta_rad (measured
    from array broadside, i.e. theta=0 means perpendicular to the
    array):
        a(theta) = exp(j * 2*pi * d/lam * sin(theta) * [0,1,...,M-1])
  - Angles to/from these functions are in DEGREES at the API
    boundary; conversion to radians is internal.
  - All numeric outputs are numpy arrays of dtype complex128 or
    float64 as appropriate. No object dtypes.

PUBLIC FUNCTIONS
----------------
  Array signal generation:
      simulate_ula_signals(...)        synthesize multichannel signals

  Geometry / steering:
      steering_vector_ula(...)         single steering vector
      array_factor(...)                |w^H a(theta)| over theta

  Beamformers (each returns weights `w` shape (M,) and the resulting
  beamformed signal `y` shape (n_snapshots,)):
      delay_and_sum(...)
      mvdr_beamformer(...)
      lcmv_beamformer(...)

  Direction-of-arrival estimators:
      music_doa(...)
      srp_phat_doa(...)

  Helpers:
      sample_covariance(rx_signals)
      angle_grid(start_deg=-90, stop_deg=90, step_deg=1)
      plot_beam_pattern(...)
      plot_doa_spectrum(...)
"""

from __future__ import annotations

import math
import functools
import numpy as np
from typing import Optional, Tuple, Sequence


# ----------------------------------------------------------------------
# Parameter-alias decorator
# ----------------------------------------------------------------------
# Small LLMs use natural parameter names ('w', 'signal', 'X') instead of
# our canonical names ('weights', 'rx_signals'). We accept both via a
# decorator that rewrites kwargs at call time.

# Map: alias_kwarg -> canonical_kwarg
# IMPORTANT: only include aliases that are NOT real parameter names of any
# public function. theta_deg, angles_deg, weights, etc. are real names so
# they must NOT appear as alias keys.
_PARAM_ALIASES = {
    # weights aliases  (real name in plot_beam_pattern, array_factor)
    "w":              "weights",
    "weight":         "weights",
    "filter":         "weights",
    "h":              "weights",
    # received signal aliases  (real name: rx_signals)
    "X":              "rx_signals",
    "x":              "rx_signals",
    "signal":         "rx_signals",
    "signals":        "rx_signals",
    "mic_signals":    "rx_signals",
    "data":           "rx_signals",
    # steering direction aliases for the BEAMFORMERS (real name: steer_deg)
    # NB: do NOT alias theta_deg here because steering_vector_ula uses
    # theta_deg as a real parameter name.
    "angle":          "steer_deg",
    "angle_deg":      "steer_deg",
    "doa":            "steer_deg",
    "doa_deg":        "steer_deg",
    "look_dir":       "steer_deg",
    "steer":          "steer_deg",
    # snr aliases (real: snr_db)
    "snr":            "snr_db",
    "SNR":            "snr_db",
    "SNR_dB":         "snr_db",
    # spacing aliases (real: d_over_lambda)
    "d_lambda":       "d_over_lambda",
    "d_l":            "d_over_lambda",
    "spacing":        "d_over_lambda",
    # number of sensors aliases (real: n_sensors)
    "M":              "n_sensors",
    "n_mics":         "n_sensors",
    "n_elements":     "n_sensors",
    "num_mics":       "n_sensors",
    "num_sensors":    "n_sensors",
    # snapshots aliases (real: n_snapshots)
    "N":              "n_snapshots",
    "n_samples":      "n_snapshots",
    "snapshots":      "n_snapshots",
    # source angles (real: source_angles_deg)
    "source_angles":  "source_angles_deg",
    "src_angles":     "source_angles_deg",
    "doas":           "source_angles_deg",
    "doas_deg":       "source_angles_deg",
    # angle grid aliases (real: angles_deg)
    "angles":         "angles_deg",
    "theta_grid":     "angles_deg",
    # plot helpers (real: spectrum)
    "spec":           "spectrum",
    "p_music":        "spectrum",
    "pseudo_spectrum":"spectrum",
    # title cosmetics (real: title)
    "label":          "title",
    "plot_title":     "title",
}


def accept_param_aliases(func):
    """Decorator: rewrite kwargs whose names match _PARAM_ALIASES into the
    canonical kwarg names. We inspect the wrapped function's signature
    so we never rename a kwarg that IS already a real parameter name of
    that function (which would break it).

    Conflict (both alias and canonical passed with different values) raises
    a clear ValueError.
    """
    import inspect
    try:
        sig = inspect.signature(func)
        real_params = set(sig.parameters.keys())
    except (TypeError, ValueError):
        real_params = set()

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        rewritten = {}
        for k, v in kwargs.items():
            # If k IS already a real parameter of func, never rewrite it.
            if k in real_params:
                rewritten[k] = v
                continue
            if k in _PARAM_ALIASES:
                canonical = _PARAM_ALIASES[k]
                # Only rewrite if the canonical name IS a real parameter
                if canonical not in real_params:
                    rewritten[k] = v
                    continue
                if canonical in kwargs and kwargs[canonical] is not v:
                    raise ValueError(
                        f"Conflict: both {k!r} (alias) and {canonical!r} "
                        f"(canonical) passed with different values."
                    )
                rewritten[canonical] = v
            else:
                rewritten[k] = v
        return func(*args, **rewritten)
    return wrapper


# ----------------------------------------------------------------------
# Tiny helpers
# ----------------------------------------------------------------------

def _check_rx(rx_signals: np.ndarray) -> Tuple[int, int]:
    """Validate the (n_sensors, n_snapshots) convention. Returns
    (n_sensors, n_snapshots)."""
    rx = np.asarray(rx_signals)
    if rx.ndim != 2:
        raise ValueError(
            f"rx_signals must be 2-D with shape (n_sensors, n_snapshots); "
            f"got shape {rx.shape}"
        )
    if rx.shape[0] > rx.shape[1]:
        # Cheap heuristic warning, not a hard error -- valid for some setups
        # but the most common bug is passing the transpose.
        # We don't raise, but the warning lives in a callable so user code
        # that triggers it can be diagnosed easily.
        pass
    return rx.shape[0], rx.shape[1]


@accept_param_aliases
def sample_covariance(rx_signals: np.ndarray) -> np.ndarray:
    """Sample spatial covariance.

    rx_signals : shape (n_sensors, n_snapshots), complex
    returns    : shape (n_sensors, n_sensors), complex
    """
    rx = np.asarray(rx_signals)
    n_sensors, n_snapshots = _check_rx(rx)
    if n_snapshots == 0:
        raise ValueError("n_snapshots must be > 0")
    return (rx @ rx.conj().T) / float(n_snapshots)


@accept_param_aliases
def angle_grid(start_deg: float = -90.0,
               stop_deg: float = 90.0,
               step_deg: float = 1.0) -> np.ndarray:
    """Convenience: angle grid in degrees, inclusive at both ends."""
    n = int(round((stop_deg - start_deg) / step_deg)) + 1
    return np.linspace(start_deg, stop_deg, n)


# ----------------------------------------------------------------------
# Steering vectors and array factors
# ----------------------------------------------------------------------

@accept_param_aliases
def steering_vector_ula(theta_deg: float,
                        n_sensors: int,
                        d_over_lambda: float = 0.5) -> np.ndarray:
    """Steering vector for a uniform linear array.

    theta_deg     : look direction relative to array broadside, in degrees
    n_sensors     : number of array elements
    d_over_lambda : element spacing in wavelengths (0.5 = half-wavelength)

    Returns array of shape (n_sensors,) complex.
    """
    if n_sensors < 1:
        raise ValueError("n_sensors must be >= 1")
    theta_rad = math.radians(theta_deg)
    k = np.arange(n_sensors)
    return np.exp(1j * 2.0 * math.pi * d_over_lambda * math.sin(theta_rad) * k)


@accept_param_aliases
def array_factor(weights: np.ndarray,
                 d_over_lambda: float = 0.5,
                 angles_deg: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the array factor |w^H a(theta)| over a sweep of angles.

    weights       : shape (n_sensors,), complex
    d_over_lambda : spacing in wavelengths
    angles_deg    : 1-D array of angles in degrees; default -90..90 step 1

    Returns:
        angles_deg : the angles used (degrees)
        magnitude  : shape (len(angles_deg),) magnitude of array factor
    """
    w = np.asarray(weights).reshape(-1).astype(np.complex128)
    n_sensors = w.size
    if angles_deg is None:
        angles_deg = angle_grid()
    angles_deg = np.asarray(angles_deg).astype(float).reshape(-1)
    # Build a steering matrix A of shape (n_sensors, n_angles)
    theta_rad = np.deg2rad(angles_deg)
    k = np.arange(n_sensors).reshape(-1, 1)        # (M, 1)
    phases = 2.0 * math.pi * d_over_lambda * np.sin(theta_rad).reshape(1, -1) * k  # (M, T)
    A = np.exp(1j * phases)                        # (M, T)
    af = w.conj() @ A                              # (T,) complex
    return angles_deg, np.abs(af)


# ----------------------------------------------------------------------
# Signal simulation
# ----------------------------------------------------------------------

@accept_param_aliases
def simulate_ula_signals(
    n_sensors: int = 6,
    n_snapshots: int = 200,
    source_angles_deg: Sequence[float] = (30.0,),
    source_powers: Optional[Sequence[float]] = None,
    snr_db: float = 20.0,
    d_over_lambda: float = 0.5,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Synthesize a multichannel ULA recording.

    Each source emits a complex Gaussian narrowband signal. The
    received signals are linearly mixed via steering vectors and
    additive complex Gaussian noise is mixed in to hit the requested
    SNR (per sensor, averaged over all sources).

    Returns rx_signals : shape (n_sensors, n_snapshots), complex.

    Robust to common LLM mistakes:
      - source_angles_deg as a scalar -> wrapped to a 1-element list
      - source_powers as a scalar -> applied to all sources
      - source_powers as None -> uniform unit power for every source
    """
    # Allow scalar source_angles_deg ("just one source at 30")
    if isinstance(source_angles_deg, (int, float)):
        source_angles_deg = [float(source_angles_deg)]
    else:
        source_angles_deg = [float(a) for a in source_angles_deg]
    n_sources = len(source_angles_deg)

    # Tolerate the most common LLM call-pattern bugs around source_powers:
    #   - missing/None -> uniform powers
    #   - scalar int/float -> apply to every source
    #   - list/tuple/array of the right length -> use as-is
    if source_powers is None:
        source_powers = [1.0] * n_sources
    elif isinstance(source_powers, (int, float)):
        source_powers = [float(source_powers)] * n_sources
    else:
        try:
            source_powers = [float(p) for p in source_powers]
        except TypeError:
            raise TypeError(
                f"source_powers must be None, a scalar, or a sequence; "
                f"got {type(source_powers).__name__}: {source_powers!r}"
            )

    if len(source_powers) != n_sources:
        raise ValueError(
            f"source_powers has length {len(source_powers)}, but "
            f"source_angles_deg has {n_sources} sources"
        )

    # Defensive sanity checks
    n_sensors = int(n_sensors)
    n_snapshots = int(n_snapshots)
    if n_sensors < 1:
        raise ValueError("n_sensors must be >= 1")
    if n_snapshots < 1:
        raise ValueError("n_snapshots must be >= 1")

    rng = np.random.default_rng(seed)

    # Source signals: (n_sources, n_snapshots), unit-variance complex Gaussian
    # then scaled by sqrt(power)
    s = (rng.standard_normal((n_sources, n_snapshots))
         + 1j * rng.standard_normal((n_sources, n_snapshots))) / math.sqrt(2.0)
    for i, p in enumerate(source_powers):
        s[i] *= math.sqrt(p)

    # Steering matrix A of shape (n_sensors, n_sources)
    A = np.column_stack([
        steering_vector_ula(theta, n_sensors, d_over_lambda)
        for theta in source_angles_deg
    ])

    # Clean received: shape (n_sensors, n_snapshots)
    x_clean = A @ s

    # Compute total signal power per sensor and add noise to hit SNR
    sig_power = float(np.mean(np.abs(x_clean) ** 2))
    if sig_power <= 0:
        sig_power = 1.0
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = math.sqrt(noise_power / 2.0) * (
        rng.standard_normal((n_sensors, n_snapshots))
        + 1j * rng.standard_normal((n_sensors, n_snapshots))
    )

    return x_clean + noise


# ----------------------------------------------------------------------
# Beamformers
# ----------------------------------------------------------------------

@accept_param_aliases
def delay_and_sum(rx_signals: np.ndarray,
                  steer_deg: float,
                  d_over_lambda: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
    """Conventional (a.k.a. Bartlett) beamformer steered to one direction.

    Returns (weights, output) where weights has shape (n_sensors,) and
    output has shape (n_snapshots,)."""
    n_sensors, _ = _check_rx(rx_signals)
    a = steering_vector_ula(steer_deg, n_sensors, d_over_lambda)
    w = a / n_sensors            # normalize so that w^H a == 1
    y = w.conj() @ rx_signals    # shape (n_snapshots,)
    return w, y


@accept_param_aliases
def mvdr_beamformer(rx_signals: np.ndarray,
                    steer_deg: float,
                    d_over_lambda: float = 0.5,
                    diagonal_loading: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Minimum-Variance Distortionless-Response beamformer.

    Solves    min  w^H R w
              s.t. w^H a(theta_s) = 1
    Closed form:
              w = R^{-1} a / (a^H R^{-1} a)

    diagonal_loading : small regularizer added to R for numerical
                       stability (set to 0 to disable).

    Returns (weights, output). weights shape (n_sensors,) complex;
    output shape (n_snapshots,) complex.
    """
    n_sensors, n_snapshots = _check_rx(rx_signals)
    R = sample_covariance(rx_signals)
    if diagonal_loading > 0:
        R = R + diagonal_loading * np.trace(R) / n_sensors * np.eye(n_sensors)
    a = steering_vector_ula(steer_deg, n_sensors, d_over_lambda)
    R_inv_a = np.linalg.solve(R, a)
    denom = np.vdot(a, R_inv_a)        # a^H @ R^{-1} @ a, scalar
    if abs(denom) < 1e-30:
        raise np.linalg.LinAlgError("MVDR denominator is ~0; check inputs")
    w = R_inv_a / denom
    y = w.conj() @ rx_signals
    return w, y


@accept_param_aliases
def lcmv_beamformer(rx_signals: np.ndarray,
                    constraint_angles_deg: Sequence[float],
                    constraint_responses: Sequence[float],
                    d_over_lambda: float = 0.5,
                    diagonal_loading: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Linearly-Constrained Minimum-Variance beamformer.

    Solves    min  w^H R w   s.t.   C^H w = f
    where C is an (n_sensors, L) constraint matrix built from the
    steering vectors of `constraint_angles_deg`, and f is
    `constraint_responses`.

    Typical use:
        - one constraint at the target angle with response 1
        - additional constraints at interferer angles with response 0

    Returns (weights, output) -- shapes as in mvdr_beamformer.
    """
    n_sensors, _ = _check_rx(rx_signals)
    if len(constraint_angles_deg) != len(constraint_responses):
        raise ValueError("constraint_angles_deg and constraint_responses must match")
    if len(constraint_angles_deg) < 1:
        raise ValueError("need at least one constraint")
    if len(constraint_angles_deg) > n_sensors:
        raise ValueError("cannot have more constraints than sensors")

    R = sample_covariance(rx_signals)
    if diagonal_loading > 0:
        R = R + diagonal_loading * np.trace(R) / n_sensors * np.eye(n_sensors)

    # Build constraint matrix
    C = np.column_stack([
        steering_vector_ula(theta, n_sensors, d_over_lambda)
        for theta in constraint_angles_deg
    ])
    f = np.asarray(constraint_responses, dtype=np.complex128).reshape(-1)

    # w = R^{-1} C ( C^H R^{-1} C )^{-1} f
    R_inv_C = np.linalg.solve(R, C)            # (M, L)
    M = C.conj().T @ R_inv_C                   # (L, L)
    w = R_inv_C @ np.linalg.solve(M, f)        # (M,)
    y = w.conj() @ rx_signals
    return w, y


# ----------------------------------------------------------------------
# DOA estimators
# ----------------------------------------------------------------------

@accept_param_aliases
def music_doa(rx_signals: np.ndarray,
              n_sources: int,
              d_over_lambda: float = 0.5,
              angles_deg: Optional[np.ndarray] = None
              ) -> Tuple[np.ndarray, np.ndarray]:
    """MUSIC direction-of-arrival spatial spectrum.

    n_sources : number of sources to assume (model order).
    Returns (angles_deg, p_music) where p_music is the pseudo-spectrum
    (peaks at estimated DOAs). Both shape (len(angles_deg),)."""
    n_sensors, n_snapshots = _check_rx(rx_signals)
    if n_sources < 1 or n_sources >= n_sensors:
        raise ValueError("require 1 <= n_sources < n_sensors")
    R = sample_covariance(rx_signals)
    # Eigendecomposition; eigenvalues sorted ascending in numpy
    eigvals, eigvecs = np.linalg.eigh(R)
    # Noise subspace = the M - K smallest eigenvalues' eigenvectors
    En = eigvecs[:, : n_sensors - n_sources]   # (M, M - K)
    if angles_deg is None:
        angles_deg = angle_grid()
    angles_deg = np.asarray(angles_deg).astype(float).reshape(-1)
    # Build steering matrix A_grid of shape (M, T)
    theta_rad = np.deg2rad(angles_deg)
    k = np.arange(n_sensors).reshape(-1, 1)
    phases = 2.0 * math.pi * d_over_lambda * np.sin(theta_rad).reshape(1, -1) * k
    A_grid = np.exp(1j * phases)                    # (M, T)
    # For each angle, denom = a^H E_n E_n^H a
    EnH_A = En.conj().T @ A_grid                    # (M-K, T)
    denom = np.sum(np.abs(EnH_A) ** 2, axis=0)      # (T,)
    # Avoid div-by-zero
    denom = np.where(denom < 1e-30, 1e-30, denom)
    p_music = 1.0 / denom
    return angles_deg, p_music


@accept_param_aliases
def srp_phat_doa(rx_signals: np.ndarray,
                 d_over_lambda: float = 0.5,
                 angles_deg: Optional[np.ndarray] = None
                 ) -> Tuple[np.ndarray, np.ndarray]:
    """SRP-PHAT spatial spectrum for narrowband signals.

    Returns (angles_deg, power). Peaks indicate source DOAs."""
    n_sensors, n_snapshots = _check_rx(rx_signals)
    # Phase-transformed covariance (normalize each entry to unit magnitude)
    R = sample_covariance(rx_signals)
    mag = np.abs(R)
    mag = np.where(mag < 1e-30, 1e-30, mag)
    R_phat = R / mag
    if angles_deg is None:
        angles_deg = angle_grid()
    angles_deg = np.asarray(angles_deg).astype(float).reshape(-1)
    theta_rad = np.deg2rad(angles_deg)
    k = np.arange(n_sensors).reshape(-1, 1)
    phases = 2.0 * math.pi * d_over_lambda * np.sin(theta_rad).reshape(1, -1) * k
    A_grid = np.exp(1j * phases)                    # (M, T)
    # power(theta) = a^H R_phat a
    out = np.einsum("mt,mn,nt->t", A_grid.conj(), R_phat, A_grid)
    # The imaginary part should be ~0 (Hermitian quadratic form)
    return angles_deg, np.real(out)


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

@accept_param_aliases
def plot_beam_pattern(weights: np.ndarray,
                      d_over_lambda: float = 0.5,
                      title: str = "Beam pattern",
                      db_floor: float = -50.0):
    """Plot |w^H a(theta)| in dB. Calls matplotlib; figure is captured
    by the sandbox runner. Returns (angles_deg, mag_db)."""
    import matplotlib.pyplot as plt
    angles, mag = array_factor(weights, d_over_lambda)
    mag = np.where(mag < 1e-12, 1e-12, mag)
    mag_db = 20.0 * np.log10(mag / np.max(mag))
    mag_db = np.maximum(mag_db, db_floor)
    plt.figure(figsize=(8, 4))
    plt.plot(angles, mag_db)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Response (dB, normalized)")
    plt.title(title)
    plt.grid(True)
    plt.ylim(db_floor, 5)
    return angles, mag_db


@accept_param_aliases
def plot_doa_spectrum(angles_deg: np.ndarray,
                      spectrum: np.ndarray,
                      title: str = "DOA spatial spectrum",
                      log_scale: bool = True):
    """Plot a DOA pseudo-spectrum. Returns (angles_deg, plotted_y)."""
    import matplotlib.pyplot as plt
    y = np.asarray(spectrum, dtype=float)
    if log_scale:
        y_pos = np.where(y > 1e-30, y, 1e-30)
        y_plot = 10.0 * np.log10(y_pos / np.max(y_pos))
    else:
        y_plot = y / np.max(y)
    plt.figure(figsize=(8, 4))
    plt.plot(angles_deg, y_plot)
    plt.xlabel("Angle (degrees)")
    plt.ylabel("Power (dB)" if log_scale else "Power (normalized)")
    plt.title(title)
    plt.grid(True)
    return angles_deg, y_plot


# ----------------------------------------------------------------------
# Self-introspection (so the LLM can be told what functions exist)
# ----------------------------------------------------------------------

PUBLIC_API = (
    # Batch 12 originals
    "sample_covariance",
    "angle_grid",
    "steering_vector_ula",
    "array_factor",
    "simulate_ula_signals",
    "delay_and_sum",
    "mvdr_beamformer",
    "lcmv_beamformer",
    "music_doa",
    "srp_phat_doa",
    "plot_beam_pattern",
    "plot_doa_spectrum",
    # Batch 12B additions
    "steering_vector_arbitrary",
    "simulate_array_signals",
    "delay_and_sum_arbitrary",
    "mvdr_arbitrary",
    "broadband_simulate",
    "broadband_delay_and_sum",
    "broadband_mvdr",
    "simulate_room_recording",
    "pesq_score",
    "stoi_score",
    "gcc_phat",
    "gcc_phat_doa_pair",
    "mic_geometry_circular",
    "mic_geometry_planar_rect",
)


# ======================================================================
# BATCH 12B EXPANSIONS
# ======================================================================
# Adds: arbitrary array geometry, broadband (STFT) processing, room
# acoustics, PESQ/STOI quality metrics, GCC-PHAT TDOA.
#
# All Batch 12B functions also pass through @accept_param_aliases for
# consistency with the Batch 12 originals.

# ----------------------------------------------------------------------
# Arbitrary array geometry
# ----------------------------------------------------------------------

SPEED_OF_SOUND = 343.0  # m/s, at ~20 deg C, dry air


@accept_param_aliases
def mic_geometry_circular(n_mics: int, radius_m: float) -> np.ndarray:
    """Return mic positions for a circular array in the XY plane.

    n_mics   : number of microphones
    radius_m : array radius in meters

    Returns positions of shape (n_mics, 2)  -- x, y in meters.
    """
    if n_mics < 1:
        raise ValueError("n_mics must be >= 1")
    if radius_m <= 0:
        raise ValueError("radius_m must be > 0")
    theta = 2.0 * math.pi * np.arange(n_mics) / n_mics
    return np.column_stack([radius_m * np.cos(theta), radius_m * np.sin(theta)])


@accept_param_aliases
def mic_geometry_planar_rect(n_rows: int, n_cols: int,
                             dx_m: float = 0.05, dy_m: float = 0.05) -> np.ndarray:
    """Return mic positions for a rectangular planar grid in the XY plane.

    Returns positions of shape (n_rows * n_cols, 2), in meters.
    """
    if n_rows < 1 or n_cols < 1:
        raise ValueError("n_rows and n_cols must be >= 1")
    xs = (np.arange(n_cols) - (n_cols - 1) / 2.0) * dx_m
    ys = (np.arange(n_rows) - (n_rows - 1) / 2.0) * dy_m
    xv, yv = np.meshgrid(xs, ys)
    return np.column_stack([xv.ravel(), yv.ravel()])


@accept_param_aliases
def steering_vector_arbitrary(theta_deg: float,
                              mic_positions: np.ndarray,
                              freq_hz: float = 1000.0,
                              c: float = SPEED_OF_SOUND,
                              phi_deg: float = 0.0) -> np.ndarray:
    """Steering vector for an arbitrary planar (2D) or volumetric (3D)
    array, for a planewave from direction (theta_deg, phi_deg) at
    frequency freq_hz.

    Conventions:
      - theta_deg is azimuth measured from +x axis, in the XY plane
      - phi_deg is elevation above XY plane (0 = horizon)
      - mic_positions shape (n_mics, 2) for planar or (n_mics, 3) for 3D
    """
    mic_positions = np.asarray(mic_positions, dtype=float)
    if mic_positions.ndim != 2 or mic_positions.shape[1] not in (2, 3):
        raise ValueError("mic_positions must be shape (n_mics, 2) or (n_mics, 3)")

    theta = math.radians(theta_deg)
    phi = math.radians(phi_deg)
    # Unit vector pointing FROM source TO array origin.
    # For a plane wave from (theta, phi), the wave vector direction is
    # (cos(phi)*cos(theta), cos(phi)*sin(theta), sin(phi))
    if mic_positions.shape[1] == 2:
        k_dir = np.array([math.cos(theta), math.sin(theta)])
    else:
        k_dir = np.array([
            math.cos(phi) * math.cos(theta),
            math.cos(phi) * math.sin(theta),
            math.sin(phi),
        ])
    # Per-mic delay (sec) relative to origin, then phase = -2*pi*f*tau
    tau = mic_positions @ k_dir / c
    return np.exp(-1j * 2.0 * math.pi * freq_hz * tau)


@accept_param_aliases
def simulate_array_signals(
    mic_positions: np.ndarray,
    n_snapshots: int = 200,
    source_angles_deg: Sequence[float] = (30.0,),
    source_powers: Optional[Sequence[float]] = None,
    snr_db: float = 20.0,
    freq_hz: float = 1000.0,
    c: float = SPEED_OF_SOUND,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Simulate received signals on an arbitrary array (planar or 3D),
    narrowband at freq_hz. Returns shape (n_mics, n_snapshots) complex.
    """
    mic_positions = np.asarray(mic_positions, dtype=float)
    n_mics = mic_positions.shape[0]

    if isinstance(source_angles_deg, (int, float)):
        source_angles_deg = [float(source_angles_deg)]
    else:
        source_angles_deg = [float(a) for a in source_angles_deg]
    n_sources = len(source_angles_deg)

    if source_powers is None:
        source_powers = [1.0] * n_sources
    elif isinstance(source_powers, (int, float)):
        source_powers = [float(source_powers)] * n_sources
    if len(source_powers) != n_sources:
        raise ValueError("source_powers must match source_angles_deg length")

    rng = np.random.default_rng(seed)
    s = (rng.standard_normal((n_sources, n_snapshots))
         + 1j * rng.standard_normal((n_sources, n_snapshots))) / math.sqrt(2.0)
    for i, p in enumerate(source_powers):
        s[i] *= math.sqrt(p)

    A = np.column_stack([
        steering_vector_arbitrary(t, mic_positions, freq_hz=freq_hz, c=c)
        for t in source_angles_deg
    ])
    x_clean = A @ s

    sig_power = float(np.mean(np.abs(x_clean) ** 2))
    if sig_power <= 0:
        sig_power = 1.0
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = math.sqrt(noise_power / 2.0) * (
        rng.standard_normal((n_mics, n_snapshots))
        + 1j * rng.standard_normal((n_mics, n_snapshots))
    )
    return x_clean + noise


@accept_param_aliases
def delay_and_sum_arbitrary(rx_signals: np.ndarray,
                            mic_positions: np.ndarray,
                            steer_deg: float,
                            freq_hz: float = 1000.0,
                            c: float = SPEED_OF_SOUND
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """Delay-and-sum beamformer for an arbitrary array."""
    rx = np.asarray(rx_signals)
    if rx.shape[0] != np.asarray(mic_positions).shape[0]:
        raise ValueError(
            f"mic count mismatch: rx_signals has {rx.shape[0]} rows, "
            f"mic_positions has {np.asarray(mic_positions).shape[0]} rows"
        )
    n_mics = rx.shape[0]
    a = steering_vector_arbitrary(steer_deg, mic_positions, freq_hz=freq_hz, c=c)
    w = a / n_mics
    y = w.conj() @ rx
    return w, y


@accept_param_aliases
def mvdr_arbitrary(rx_signals: np.ndarray,
                   mic_positions: np.ndarray,
                   steer_deg: float,
                   freq_hz: float = 1000.0,
                   c: float = SPEED_OF_SOUND,
                   diagonal_loading: float = 1e-6
                   ) -> Tuple[np.ndarray, np.ndarray]:
    """MVDR beamformer for an arbitrary array."""
    rx = np.asarray(rx_signals)
    if rx.shape[0] != np.asarray(mic_positions).shape[0]:
        raise ValueError("mic count mismatch")
    n_mics = rx.shape[0]
    R = sample_covariance(rx)
    if diagonal_loading > 0:
        R = R + diagonal_loading * np.trace(R) / n_mics * np.eye(n_mics)
    a = steering_vector_arbitrary(steer_deg, mic_positions, freq_hz=freq_hz, c=c)
    R_inv_a = np.linalg.solve(R, a)
    denom = np.vdot(a, R_inv_a)
    if abs(denom) < 1e-30:
        raise np.linalg.LinAlgError("MVDR denominator is ~0")
    w = R_inv_a / denom
    y = w.conj() @ rx
    return w, y


# ----------------------------------------------------------------------
# Broadband (STFT) processing
# ----------------------------------------------------------------------

def _stft(x: np.ndarray, n_fft: int = 512, hop: int = 256,
          fs: float = 16000.0) -> Tuple[np.ndarray, np.ndarray]:
    """Per-channel STFT. x shape (n_mics, n_samples) real.
    Returns:
        X : shape (n_mics, n_freqs, n_frames) complex
        freqs : shape (n_freqs,) in Hz
    """
    from scipy.signal import stft as _scipy_stft
    x = np.asarray(x, dtype=float)
    if x.ndim != 2:
        raise ValueError("input must be shape (n_mics, n_samples)")
    Xs = []
    freqs = None
    for m in range(x.shape[0]):
        f, _t, Zxx = _scipy_stft(x[m], fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                                  return_onesided=True)
        Xs.append(Zxx)
        freqs = f
    return np.stack(Xs, axis=0), freqs


def _istft(X: np.ndarray, n_fft: int = 512, hop: int = 256,
           fs: float = 16000.0, n_samples: Optional[int] = None) -> np.ndarray:
    """Inverse STFT for a single channel. X shape (n_freqs, n_frames) complex.
    Returns shape (n_samples,) real."""
    from scipy.signal import istft as _scipy_istft
    _t, y = _scipy_istft(X, fs=fs, nperseg=n_fft, noverlap=n_fft - hop,
                         input_onesided=True)
    if n_samples is not None and y.size > n_samples:
        y = y[:n_samples]
    return np.real(y)


@accept_param_aliases
def broadband_simulate(
    mic_positions: np.ndarray,
    n_samples: int = 16000,
    fs: float = 16000.0,
    source_angles_deg: Sequence[float] = (30.0,),
    snr_db: float = 20.0,
    bandwidth_hz: Tuple[float, float] = (300.0, 3400.0),
    c: float = SPEED_OF_SOUND,
    seed: Optional[int] = 0,
) -> np.ndarray:
    """Simulate broadband (real-valued) array recording.

    Sources emit band-limited white noise (Gaussian) inside bandwidth_hz.
    Per-channel signals are constructed in the time domain via per-frequency
    steering vectors (so it works for arbitrary geometry).

    Returns:
        x : shape (n_mics, n_samples), real-valued time-domain signals.
    """
    mic_positions = np.asarray(mic_positions, dtype=float)
    n_mics = mic_positions.shape[0]
    if isinstance(source_angles_deg, (int, float)):
        source_angles_deg = [float(source_angles_deg)]
    else:
        source_angles_deg = [float(a) for a in source_angles_deg]
    n_sources = len(source_angles_deg)
    rng = np.random.default_rng(seed)

    # Build each source as bandlimited noise
    # Apply per-mic delay via STFT (per-frequency steering)
    # Simpler implementation: build s, then for each frequency bin apply
    # steering vector, recombine via ISTFT.
    s = rng.standard_normal((n_sources, n_samples))
    # Apply band-limit
    from scipy.signal import butter, sosfilt
    low, high = bandwidth_hz
    nyq = fs / 2.0
    sos = butter(8, [low / nyq, min(high, nyq * 0.99) / nyq],
                 btype="bandpass", output="sos")
    for i in range(n_sources):
        s[i] = sosfilt(sos, s[i])

    # STFT each source (treat each source as a 1-mic input here)
    n_fft = 512
    hop = 256
    s_freq_list = []
    for i in range(n_sources):
        S_i, freqs = _stft(s[i:i+1], n_fft=n_fft, hop=hop, fs=fs)
        s_freq_list.append(S_i[0])  # (n_freqs, n_frames)
    s_freq = np.stack(s_freq_list, axis=0)  # (n_sources, n_freqs, n_frames)

    # Build per-frequency steering at each mic
    n_freqs = s_freq.shape[1]
    n_frames = s_freq.shape[2]
    x_freq = np.zeros((n_mics, n_freqs, n_frames), dtype=complex)
    for k_idx, fk in enumerate(freqs):
        if fk < low or fk > high or fk <= 0:
            continue
        # Per-source steering at this freq
        for i, ang in enumerate(source_angles_deg):
            a = steering_vector_arbitrary(ang, mic_positions, freq_hz=fk, c=c)
            x_freq[:, k_idx, :] += np.outer(a, s_freq[i, k_idx, :])

    # ISTFT each mic
    x = np.zeros((n_mics, n_samples))
    for m in range(n_mics):
        ym = _istft(x_freq[m], n_fft=n_fft, hop=hop, fs=fs, n_samples=n_samples)
        x[m, :len(ym)] = ym[:n_samples]

    # Add noise to hit SNR
    sig_power = float(np.mean(x ** 2))
    if sig_power <= 0:
        sig_power = 1.0
    noise_power = sig_power / (10.0 ** (snr_db / 10.0))
    noise = math.sqrt(noise_power) * rng.standard_normal((n_mics, n_samples))
    return x + noise


@accept_param_aliases
def broadband_delay_and_sum(rx_signals_time: np.ndarray,
                            mic_positions: np.ndarray,
                            steer_deg: float,
                            fs: float = 16000.0,
                            c: float = SPEED_OF_SOUND,
                            n_fft: int = 512,
                            hop: int = 256) -> np.ndarray:
    """Broadband delay-and-sum beamformer in STFT domain.
    Returns single-channel beamformed time-domain signal.
    """
    rx_signals_time = np.asarray(rx_signals_time, dtype=float)
    n_mics, n_samples = rx_signals_time.shape
    X, freqs = _stft(rx_signals_time, n_fft=n_fft, hop=hop, fs=fs)
    Y = np.zeros((X.shape[1], X.shape[2]), dtype=complex)
    for k_idx, fk in enumerate(freqs):
        if fk <= 0:
            continue
        a = steering_vector_arbitrary(steer_deg, mic_positions,
                                      freq_hz=fk, c=c)
        w = a / n_mics
        Y[k_idx, :] = w.conj() @ X[:, k_idx, :]
    y = _istft(Y, n_fft=n_fft, hop=hop, fs=fs, n_samples=n_samples)
    return y


@accept_param_aliases
def broadband_mvdr(rx_signals_time: np.ndarray,
                   mic_positions: np.ndarray,
                   steer_deg: float,
                   fs: float = 16000.0,
                   c: float = SPEED_OF_SOUND,
                   n_fft: int = 512,
                   hop: int = 256,
                   diagonal_loading: float = 1e-3) -> np.ndarray:
    """Broadband MVDR via per-frequency MVDR weights, applied in STFT.
    Returns single-channel time-domain output.
    """
    rx_signals_time = np.asarray(rx_signals_time, dtype=float)
    n_mics, n_samples = rx_signals_time.shape
    X, freqs = _stft(rx_signals_time, n_fft=n_fft, hop=hop, fs=fs)
    n_freqs, n_frames = X.shape[1], X.shape[2]
    Y = np.zeros((n_freqs, n_frames), dtype=complex)
    for k_idx, fk in enumerate(freqs):
        if fk <= 0:
            continue
        Xk = X[:, k_idx, :]                            # (n_mics, n_frames)
        Rk = (Xk @ Xk.conj().T) / max(n_frames, 1)
        Rk = Rk + diagonal_loading * np.trace(Rk) / n_mics * np.eye(n_mics)
        a = steering_vector_arbitrary(steer_deg, mic_positions,
                                      freq_hz=fk, c=c)
        try:
            R_inv_a = np.linalg.solve(Rk, a)
            denom = np.vdot(a, R_inv_a)
            if abs(denom) < 1e-30:
                w = a / n_mics
            else:
                w = R_inv_a / denom
        except np.linalg.LinAlgError:
            w = a / n_mics
        Y[k_idx, :] = w.conj() @ Xk
    return _istft(Y, n_fft=n_fft, hop=hop, fs=fs, n_samples=n_samples)


# ----------------------------------------------------------------------
# Room acoustics  (via pyroomacoustics)
# ----------------------------------------------------------------------

@accept_param_aliases
def simulate_room_recording(room_dims_m: Sequence[float],
                            mic_positions: np.ndarray,
                            src_positions: np.ndarray,
                            src_signals: np.ndarray,
                            rt60_sec: float = 0.4,
                            fs: float = 16000.0,
                            ) -> np.ndarray:
    """Simulate a multichannel room recording with reverberation using
    pyroomacoustics.

    Args:
        room_dims_m  : (3,) tuple/list [Lx, Ly, Lz]  -- room dimensions in meters
        mic_positions : shape (n_mics, 3)            -- mic positions in meters
        src_positions : shape (n_sources, 3)         -- source positions in meters
        src_signals   : shape (n_sources, n_samples) -- source time-domain signals
        rt60_sec      : reverberation time (seconds)
        fs            : sample rate in Hz

    Returns:
        rec : shape (n_mics, n_samples_out) real-valued mic signals

    Notes:
        - Requires `pyroomacoustics` package. Install with:
              pip install pyroomacoustics
        - Output length is slightly longer than n_samples due to convolution.
    """
    try:
        import pyroomacoustics as pra
    except ImportError:
        raise ImportError(
            "simulate_room_recording requires pyroomacoustics. "
            "Install with: pip install pyroomacoustics"
        )
    room_dims_m = list(map(float, room_dims_m))
    if len(room_dims_m) != 3:
        raise ValueError("room_dims_m must be 3-D [Lx, Ly, Lz]")
    mic_positions = np.asarray(mic_positions, dtype=float)
    src_positions = np.asarray(src_positions, dtype=float)
    src_signals = np.asarray(src_signals, dtype=float)
    if src_signals.ndim == 1:
        src_signals = src_signals[None, :]
    n_sources = src_positions.shape[0]
    if src_signals.shape[0] != n_sources:
        raise ValueError(
            f"src_signals has {src_signals.shape[0]} rows but "
            f"src_positions has {n_sources} sources"
        )
    if mic_positions.shape[1] != 3:
        raise ValueError("mic_positions must be (n_mics, 3)")
    if src_positions.shape[1] != 3:
        raise ValueError("src_positions must be (n_sources, 3)")

    e_absorption, max_order = pra.inverse_sabine(rt60_sec, room_dims_m)
    room = pra.ShoeBox(room_dims_m, fs=fs,
                       materials=pra.Material(e_absorption),
                       max_order=int(max_order))
    # Sources
    for i in range(n_sources):
        room.add_source(src_positions[i].tolist(), signal=src_signals[i])
    # Microphones
    room.add_microphone_array(mic_positions.T)
    room.simulate()
    # room.mic_array.signals shape: (n_mics, n_out_samples)
    return np.asarray(room.mic_array.signals, dtype=float)


# ----------------------------------------------------------------------
# Speech-quality metrics  (PESQ + STOI)
# ----------------------------------------------------------------------

@accept_param_aliases
def pesq_score(clean: np.ndarray,
               noisy: np.ndarray,
               fs: float = 16000.0,
               mode: str = "wb") -> float:
    """Perceptual Evaluation of Speech Quality (PESQ).
    mode: 'wb' (wideband, fs=16k) or 'nb' (narrowband, fs=8k).
    Returns PESQ score in approximately [-0.5, 4.5].
    """
    try:
        from pesq import pesq as _pesq
    except ImportError:
        raise ImportError(
            "pesq_score requires the `pesq` package. Install with: pip install pesq"
        )
    clean = np.asarray(clean, dtype=float).flatten()
    noisy = np.asarray(noisy, dtype=float).flatten()
    n = min(len(clean), len(noisy))
    return float(_pesq(int(fs), clean[:n], noisy[:n], mode))


@accept_param_aliases
def stoi_score(clean: np.ndarray,
               noisy: np.ndarray,
               fs: float = 16000.0,
               extended: bool = False) -> float:
    """Short-Time Objective Intelligibility (STOI). Returns score in [0, 1].
    If `extended=True`, returns the ESTOI variant.
    """
    try:
        from pystoi import stoi as _stoi
    except ImportError:
        raise ImportError(
            "stoi_score requires the `pystoi` package. Install with: pip install pystoi"
        )
    clean = np.asarray(clean, dtype=float).flatten()
    noisy = np.asarray(noisy, dtype=float).flatten()
    n = min(len(clean), len(noisy))
    return float(_stoi(clean[:n], noisy[:n], int(fs), extended=extended))


# ----------------------------------------------------------------------
# GCC-PHAT
# ----------------------------------------------------------------------

@accept_param_aliases
def gcc_phat(x1: np.ndarray, x2: np.ndarray,
             fs: float = 16000.0,
             max_tau_sec: Optional[float] = None,
             interp: int = 1
             ) -> Tuple[float, np.ndarray, np.ndarray]:
    """Generalized cross-correlation with phase transform.

    x1, x2 : 1-D real signals
    fs     : sample rate
    max_tau_sec : if set, the lag is restricted to +/- max_tau_sec
    interp : zero-padding factor for the FFT. >1 = more linear-correlation
             padding (helps when signals are short). Does NOT do
             fractional-sample upsampling.

    Returns:
        tau_sec  : estimated TDOA in seconds, with PARABOLIC SUB-SAMPLE
                   INTERPOLATION around the peak for fractional accuracy.
                   Positive => x2 arrives later than x1.
        lags_sec : the lag axis (seconds), at original sample resolution
        gcc      : the GCC-PHAT correlation over `lags_sec`
    """
    x1 = np.asarray(x1, dtype=float).flatten()
    x2 = np.asarray(x2, dtype=float).flatten()
    n = max(len(x1), len(x2))
    n_fft = int(2 ** np.ceil(np.log2(2 * n))) * max(1, int(interp))
    X1 = np.fft.rfft(x1, n=n_fft)
    X2 = np.fft.rfft(x2, n=n_fft)
    R = X1 * np.conj(X2)
    mag = np.abs(R)
    mag = np.where(mag < 1e-30, 1e-30, mag)
    Rphat = R / mag
    cc_raw = np.fft.irfft(Rphat, n=n_fft)
    # cc_raw[k] is the correlation at original-rate lag of `k` samples
    # (with wrap-around for negative lags via -k mod n_fft).
    # Center the array so index 0 is lag 0.
    max_shift = n_fft // 2
    cc = np.concatenate((cc_raw[-max_shift:], cc_raw[:max_shift]))
    lag_samples = np.arange(-max_shift, max_shift)
    lags_sec = lag_samples / fs
    if max_tau_sec is not None:
        within = np.abs(lags_sec) <= max_tau_sec
        cc = cc[within]
        lags_sec = lags_sec[within]
        lag_samples = lag_samples[within]
    peak = int(np.argmax(np.abs(cc)))
    # Parabolic interpolation around the peak for sub-sample accuracy
    if 1 <= peak <= len(cc) - 2:
        y0, y1, y2 = cc[peak - 1], cc[peak], cc[peak + 1]
        denom = (y0 - 2.0 * y1 + y2)
        if abs(denom) > 1e-30:
            offset = 0.5 * (y0 - y2) / denom
            offset = max(-1.0, min(1.0, offset))
        else:
            offset = 0.0
        tau_sec = (lag_samples[peak] + offset) / fs
    else:
        tau_sec = float(lags_sec[peak])
    return tau_sec, lags_sec, cc


@accept_param_aliases
def gcc_phat_doa_pair(mic1_signal: np.ndarray,
                      mic2_signal: np.ndarray,
                      mic_spacing_m: float,
                      fs: float = 16000.0,
                      c: float = SPEED_OF_SOUND
                      ) -> float:
    """Estimate DOA (degrees from broadside) of a planewave source from
    a single mic pair using GCC-PHAT.

    Convention: positive DOA means the source is closer to mic 2 than to
    mic 1 (so mic 2 receives the wavefront EARLIER, mic 1 LATER, and the
    cross-correlation peak of (mic1, mic2) is at a NEGATIVE lag).

    Returns the estimated DOA in degrees in [-90, 90].
    """
    max_tau = mic_spacing_m / c
    tau_sec, _lags, _cc = gcc_phat(mic1_signal, mic2_signal, fs=fs,
                                   max_tau_sec=max_tau)
    # Flip sign so positive theta means source nearer mic2.
    # tau < 0 => mic2 sees source first => source toward +d direction => +theta
    arg = max(-1.0, min(1.0, -c * tau_sec / mic_spacing_m))
    return math.degrees(math.asin(arg))


# ----------------------------------------------------------------------
# Aliases for common LLM typos / paraphrased names
# (kept from earlier hotfix)
# ----------------------------------------------------------------------

mvdr_beamform = mvdr_beamformer
mvdr = mvdr_beamformer
delay_sum = delay_and_sum
das = delay_and_sum
lcmv = lcmv_beamformer
music = music_doa
srp_phat = srp_phat_doa
steering_vector = steering_vector_ula
ula_steering_vector = steering_vector_ula
beam_pattern = array_factor
covariance = sample_covariance
sample_cov = sample_covariance
simulate_signals = simulate_ula_signals
simulate_array = simulate_array_signals
plot_beampattern = plot_beam_pattern
plot_spectrum = plot_doa_spectrum
# Batch 12B aliases
simulate_room = simulate_room_recording
pesq = pesq_score
stoi = stoi_score
broadband_das = broadband_delay_and_sum


def describe_api() -> str:
    """Return a human-readable summary of the toolkit. Used to inject
    into the LLM system prompt so it knows what's available."""
    return (
        "Available helpers in `dsp_toolkit` (already imported in the sandbox).\n"
        "ALWAYS pass arguments after the first two as KEYWORDS to avoid mistakes.\n\n"
        "=== ULA (narrowband) ===\n"
        "  simulate_ula_signals(n_sensors, n_snapshots,\n"
        "                       source_angles_deg=[30.0], snr_db=20.0,\n"
        "                       d_over_lambda=0.5, seed=0)\n"
        "    -> X with shape (n_sensors, n_snapshots) complex\n"
        "  steering_vector_ula(theta_deg, n_sensors, d_over_lambda=0.5) -> (M,) complex\n"
        "  array_factor(weights, d_over_lambda=0.5, angles_deg=None) -> (angles, |AF|)\n"
        "  sample_covariance(rx_signals) -> (M, M); rx_signals shape (M, N)\n"
        "  delay_and_sum(rx_signals, steer_deg) -> (weights, output)\n"
        "  mvdr_beamformer(rx_signals, steer_deg, diagonal_loading=1e-6) -> (weights, output)\n"
        "  lcmv_beamformer(rx_signals, constraint_angles_deg, constraint_responses) -> (w, y)\n"
        "  music_doa(rx_signals, n_sources, angles_deg=None) -> (angles, spectrum)\n"
        "  srp_phat_doa(rx_signals, angles_deg=None) -> (angles, spectrum)\n"
        "  plot_beam_pattern(weights, title='...', db_floor=-50)\n"
        "  plot_doa_spectrum(angles_deg, spectrum, title='...', log_scale=True)\n\n"
        "=== ARBITRARY ARRAY GEOMETRY (Batch 12B) ===\n"
        "  mic_geometry_circular(n_mics, radius_m) -> (n_mics, 2)\n"
        "  mic_geometry_planar_rect(n_rows, n_cols, dx_m=0.05, dy_m=0.05) -> (n_mics, 2)\n"
        "  steering_vector_arbitrary(theta_deg, mic_positions, freq_hz=1000)\n"
        "    -> (n_mics,) complex; mic_positions shape (n_mics, 2 or 3)\n"
        "  simulate_array_signals(mic_positions, n_snapshots=200,\n"
        "                          source_angles_deg=[30], snr_db=20.0,\n"
        "                          freq_hz=1000) -> (n_mics, n_snapshots)\n"
        "  delay_and_sum_arbitrary(rx, mic_positions, steer_deg, freq_hz=1000) -> (w, y)\n"
        "  mvdr_arbitrary(rx, mic_positions, steer_deg, freq_hz=1000) -> (w, y)\n\n"
        "=== BROADBAND (STFT, Batch 12B) ===\n"
        "  broadband_simulate(mic_positions, n_samples=16000, fs=16000,\n"
        "                     source_angles_deg=[30], snr_db=20,\n"
        "                     bandwidth_hz=(300, 3400)) -> (n_mics, n_samples) REAL\n"
        "  broadband_delay_and_sum(rx_time, mic_positions, steer_deg,\n"
        "                          fs=16000) -> (n_samples,) REAL\n"
        "  broadband_mvdr(rx_time, mic_positions, steer_deg, fs=16000) -> (n_samples,) REAL\n\n"
        "=== ROOM ACOUSTICS (Batch 12B) ===\n"
        "  simulate_room_recording(room_dims_m, mic_positions_3d, src_positions_3d,\n"
        "                          src_signals, rt60_sec=0.4, fs=16000)\n"
        "    -> (n_mics, n_out_samples) REAL\n"
        "    NOTE: requires pyroomacoustics. Install: pip install pyroomacoustics\n\n"
        "=== QUALITY METRICS (Batch 12B) ===\n"
        "  pesq_score(clean, noisy, fs=16000, mode='wb') -> float in [-0.5, 4.5]\n"
        "  stoi_score(clean, noisy, fs=16000, extended=False) -> float in [0, 1]\n"
        "    NOTE: pesq needs `pesq` package; stoi needs `pystoi`.\n\n"
        "=== GCC-PHAT / TDOA (Batch 12B) ===\n"
        "  gcc_phat(x1, x2, fs=16000, max_tau_sec=None) -> (tau_sec, lags, gcc)\n"
        "  gcc_phat_doa_pair(mic1_signal, mic2_signal, mic_spacing_m, fs=16000)\n"
        "    -> doa_deg (in [-90, 90])\n\n"
        "CONVENTIONS:\n"
        "  - All time-domain signals: shape (n_mics, n_samples) REAL\n"
        "  - All narrowband ULA signals: shape (n_mics, n_snapshots) COMPLEX\n"
        "  - mic_positions: shape (n_mics, 2) for planar or (n_mics, 3) for 3D\n"
        "  - Angles in degrees; 0 deg = broadside (ULA) or +x axis (arbitrary)\n\n"
        "WORKED EXAMPLES:\n"
        "  # Narrowband ULA MVDR:\n"
        "  X = simulate_ula_signals(n_sensors=6, n_snapshots=500,\n"
        "                           source_angles_deg=[30, 60], snr_db=20)\n"
        "  w, _ = mvdr_beamformer(X, steer_deg=30)\n"
        "  plot_beam_pattern(w, title='MVDR target 30 / interferer 60')\n\n"
        "  # Circular array, narrowband MVDR:\n"
        "  mics = mic_geometry_circular(n_mics=6, radius_m=0.05)\n"
        "  X = simulate_array_signals(mics, n_snapshots=500,\n"
        "                              source_angles_deg=[30, 60], snr_db=20,\n"
        "                              freq_hz=2000)\n"
        "  w, _ = mvdr_arbitrary(X, mics, steer_deg=30, freq_hz=2000)\n\n"
        "  # Broadband (real audio):\n"
        "  mics = mic_geometry_circular(n_mics=4, radius_m=0.05)\n"
        "  x = broadband_simulate(mics, n_samples=16000, fs=16000,\n"
        "                          source_angles_deg=[45], snr_db=15)\n"
        "  y = broadband_mvdr(x, mics, steer_deg=45, fs=16000)\n"
    )

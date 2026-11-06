"""DMRS pilot-based channel estimation module."""

import numpy as np


def ls_channel_estimate(
    received_pilots: np.ndarray,
    reference_pilots: np.ndarray,
) -> np.ndarray:
    """Estimate complex channel coefficients using LS estimation.

    Args:
        received_pilots: Received complex pilot symbols.
            Shape must match reference_pilots.
        reference_pilots: Known transmitted pilot symbols.
            All pilot values must be nonzero.

    Returns:
        Complex channel estimates with the same shape as
        the input pilot arrays, calculated as Y / X.

    Raises:
        ValueError: If pilot shapes differ, arrays are empty,
            or reference pilots contain zeros.
    """
    received_pilots = np.asarray(received_pilots)
    reference_pilots = np.asarray(reference_pilots)

    if received_pilots.shape != reference_pilots.shape:
        raise ValueError("Pilot shapes must match")

    if received_pilots.size == 0:
        raise ValueError("Pilot arrays cannot be empty")

    if np.any(reference_pilots == 0):
        raise ValueError("Reference pilots cannot contain zeros")

    return received_pilots / reference_pilots


def interpolate_channel(
    pilot_estimates: np.ndarray,
    pilot_indices: np.ndarray,
    num_subcarriers: int,
) -> np.ndarray:
    """Interpolate complex channel estimates across subcarriers.

    Real and imaginary components are interpolated separately
    using one-dimensional linear interpolation. Values outside
    the pilot range use the nearest endpoint estimate.

    Args:
        pilot_estimates: One-dimensional complex channel
            estimates at known pilot positions, shape (P,).
        pilot_indices: Sorted, strictly increasing integer
            pilot positions, shape (P,), where P >= 2.
        num_subcarriers: Total number of frequency-domain
            subcarriers (K).

    Returns:
        Interpolated complex channel array with shape (K,).

    Raises:
        ValueError: If pilot dimensions are invalid, fewer
            than two pilots are provided, or pilot indices
            are unsorted, duplicated, or out of range.
    """
    pilot_estimates = np.asarray(pilot_estimates)
    pilot_indices = np.asarray(pilot_indices)

    if (
        pilot_estimates.ndim != 1
        or pilot_indices.ndim != 1
        or pilot_estimates.size != pilot_indices.size
        or pilot_indices.size < 2
    ):
        raise ValueError("Invalid pilot dimensions")

    if (
        not np.issubdtype(pilot_indices.dtype, np.integer)
        or np.any(np.diff(pilot_indices) <= 0)
        or pilot_indices[0] < 0
        or pilot_indices[-1] >= num_subcarriers
    ):
        raise ValueError("Invalid pilot indices")

    subcarriers = np.arange(num_subcarriers)

    real = np.interp(subcarriers, pilot_indices, pilot_estimates.real)
    imag = np.interp(subcarriers, pilot_indices, pilot_estimates.imag)

    return real + 1j * imag

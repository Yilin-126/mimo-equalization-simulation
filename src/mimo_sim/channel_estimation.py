"""DMRS信道估计模块。"""

import numpy as np


def ls_channel_estimate(
    received_pilots: np.ndarray,
    reference_pilots: np.ndarray,
) -> np.ndarray:
    """Estimate channel coefficients using LS pilots."""
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
    """Interpolate complex channel estimates across subcarriers."""
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

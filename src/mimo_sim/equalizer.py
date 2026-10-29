"""ZF and MMSE MIMO equalization."""

import numpy as np


def validate_inputs(
    received_signal: np.ndarray,
    channel_matrix: np.ndarray,
) -> None:
    """Validate MIMO input dimensions."""
    if received_signal.ndim != 1 or channel_matrix.ndim != 2:
        raise ValueError("Invalid input dimensions")

    num_rx, num_tx = channel_matrix.shape

    if num_tx == 0 or num_rx < num_tx:
        raise ValueError("Invalid antenna configuration")

    if received_signal.shape != (num_rx,):
        raise ValueError("Received signal shape mismatch")


def zf_equalize(
    received_signal: np.ndarray,
    channel_matrix: np.ndarray,
) -> np.ndarray:
    """Perform zero-forcing equalization."""
    validate_inputs(received_signal, channel_matrix)

    if np.linalg.matrix_rank(channel_matrix) < channel_matrix.shape[1]:
        raise ValueError("Channel matrix is rank deficient")

    return np.linalg.lstsq(channel_matrix, received_signal, rcond=None)[0]


def mmse_equalize(
    received_signal: np.ndarray,
    channel_matrix: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Perform MMSE equalization with unit signal covariance."""
    validate_inputs(received_signal, channel_matrix)

    if not np.isfinite(noise_variance) or noise_variance < 0:
        raise ValueError("Invalid noise variance")

    if noise_variance == 0:
        return zf_equalize(received_signal, channel_matrix)

    hermitian = channel_matrix.conj().T

    gram = hermitian @ channel_matrix
    regularized = gram + noise_variance * np.eye(channel_matrix.shape[1])

    return np.linalg.solve(
        regularized,
        hermitian @ received_signal,
    )

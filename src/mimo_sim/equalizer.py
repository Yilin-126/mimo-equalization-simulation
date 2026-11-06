"""ZF and MMSE MIMO equalization."""

import numpy as np


def validate_inputs(
    received_signal: np.ndarray,
    channel_matrix: np.ndarray,
) -> None:
    """Validate MIMO signal and channel matrix dimensions.

    Args:
        received_signal: Received complex signal vector
            with shape (Nr,).
        channel_matrix: Complex MIMO channel matrix
            with shape (Nr, Nt), where Nr >= Nt.

    Returns:
        None. Validation completes without returning a value.

    Raises:
        ValueError: If input dimensions are invalid, the
            antenna configuration is unsupported, or the
            received signal length does not match Nr.
    """
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
    """Recover transmitted symbols using zero-forcing equalization.

    ZF estimates the transmitted signal by solving the
    MIMO linear system using a least-squares solution.

    Args:
        received_signal: Received complex signal vector
            with shape (Nr,).
        channel_matrix: Complex MIMO channel matrix
            with shape (Nr, Nt).

    Returns:
        Estimated transmitted complex symbol vector
        with shape (Nt,).

    Raises:
        ValueError: If input dimensions are invalid or
            the channel matrix is rank deficient.
    """
    validate_inputs(received_signal, channel_matrix)

    if np.linalg.matrix_rank(channel_matrix) < channel_matrix.shape[1]:
        raise ValueError("Channel matrix is rank deficient")

    return np.linalg.lstsq(channel_matrix, received_signal, rcond=None)[0]


def mmse_equalize(
    received_signal: np.ndarray,
    channel_matrix: np.ndarray,
    noise_variance: float,
) -> np.ndarray:
    """Recover transmitted symbols using MMSE equalization.

    The equalizer applies noise-dependent regularization,
    assuming unit transmit symbol covariance and spatially
    white complex Gaussian noise.

    Args:
        received_signal: Received complex signal vector
            with shape (Nr,).
        channel_matrix: Complex MIMO channel matrix
            with shape (Nr, Nt).
        noise_variance: Nonnegative complex noise variance
            per receive antenna.

    Returns:
        Estimated transmitted complex symbol vector
        with shape (Nt,).

    Raises:
        ValueError: If input dimensions or noise variance
            are invalid, or the zero-noise channel is
            rank deficient.
        numpy.linalg.LinAlgError: If the regularized linear
            system cannot be solved.
    """
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

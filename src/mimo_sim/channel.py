"""MIMO channel models."""

import numpy as np


def add_awgn(
    signal: np.ndarray,
    noise_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex additive white Gaussian noise to a signal.

    Args:
        signal: Input complex signal array of arbitrary shape.
        noise_variance: Nonnegative complex noise variance per sample.
        rng: NumPy random generator.

    Returns:
        Noisy complex array with the same shape as the input.

    Raises:
        TypeError: If the input signal is not complex.
        ValueError: If the signal is empty or noise variance is negative.
    """
    if not np.iscomplexobj(signal):
        raise TypeError("Signal must be complex")

    if signal.size == 0 or noise_variance < 0:
        raise ValueError("Invalid signal or noise variance")

    noise = np.sqrt(noise_variance / 2) * (
        rng.standard_normal(signal.shape) + 1j * rng.standard_normal(signal.shape)
    )

    return signal + noise


def generate_mimo_channel(
    num_tx: int,
    num_rx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate an independent Rayleigh fading MIMO channel matrix.

    Args:
        num_tx: Number of transmit antennas (Nt).
        num_rx: Number of receive antennas (Nr).
        rng: NumPy random generator.

    Returns:
        Complex channel matrix with shape (Nr, Nt).
        Each coefficient follows CN(0, 1).

    Raises:
        ValueError: If either antenna count is not positive.
    """
    if num_tx <= 0 or num_rx <= 0:
        raise ValueError("Antenna counts must be positive")

    shape = (num_rx, num_tx)

    real = rng.standard_normal(shape)
    imag = rng.standard_normal(shape)

    return (real + 1j * imag) / np.sqrt(2)


def generate_frequency_channel(
    num_symbols: int,
    num_subcarriers: int,
    num_tx: int,
    num_rx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a frequency-flat MIMO channel over an OFDM grid.

    Args:
        num_symbols: Number of OFDM symbols (S).
        num_subcarriers: Number of subcarriers (K).
        num_tx: Number of transmit antennas (Nt).
        num_rx: Number of receive antennas (Nr).
        rng: NumPy random generator.

    Returns:
        Complex channel array with shape (S, K, Nr, Nt).
        The same MIMO channel matrix is repeated over all
        OFDM symbols and subcarriers.

    Raises:
        ValueError: If any dimension is not positive.
    """
    if min(num_symbols, num_subcarriers, num_tx, num_rx) <= 0:
        raise ValueError("All dimensions must be positive")

    channel = generate_mimo_channel(num_tx, num_rx, rng)

    return np.broadcast_to(
        channel,
        (num_symbols, num_subcarriers, num_rx, num_tx),
    ).copy()

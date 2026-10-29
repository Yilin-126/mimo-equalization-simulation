"""MIMO信道模型。"""

import numpy as np


def add_awgn(
    signal: np.ndarray,
    noise_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add complex AWGN with specified noise variance."""
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
    """Generate an i.i.d. Rayleigh MIMO channel matrix."""
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
    """Generate a frequency-flat MIMO channel over an OFDM grid."""
    if min(num_symbols, num_subcarriers, num_tx, num_rx) <= 0:
        raise ValueError("All dimensions must be positive")

    channel = generate_mimo_channel(num_tx, num_rx, rng)

    return np.broadcast_to(
        channel,
        (num_symbols, num_subcarriers, num_rx, num_tx),
    ).copy()

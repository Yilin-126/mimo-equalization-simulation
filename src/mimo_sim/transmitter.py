"""MIMO-OFDM transmitter."""

import numpy as np


def generate_bits(
    num_bits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate random bits."""
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    return rng.integers(0, 2, size=num_bits, dtype=np.int8)


def modulate(
    bits: np.ndarray,
    modulation_type: str,
) -> np.ndarray:
    """Perform normalized QPSK or 16QAM modulation."""
    bits = np.asarray(bits)

    bits_per_symbol = {
        "QPSK": 2,
        "16QAM": 4,
    }

    if modulation_type not in bits_per_symbol:
        raise ValueError("Unsupported modulation")

    n_bits = bits_per_symbol[modulation_type]

    if (
        bits.ndim != 1
        or len(bits) == 0
        or len(bits) % n_bits != 0
        or not np.all((bits == 0) | (bits == 1))
    ):
        raise ValueError("Invalid bit sequence")

    groups = bits.reshape(-1, n_bits).astype(np.int8)

    if modulation_type == "QPSK":
        real = 1 - 2 * groups[:, 0]
        imag = 1 - 2 * groups[:, 1]
        return (real + 1j * imag) / np.sqrt(2)

    real = (1 - 2 * groups[:, 0]) * (3 - 2 * groups[:, 1])
    imag = (1 - 2 * groups[:, 2]) * (3 - 2 * groups[:, 3])

    return (real + 1j * imag) / np.sqrt(10)


def ofdm_modulate(
    resource_grid: np.ndarray,
    cp_length: int,
) -> np.ndarray:
    """OFDM modulation interface."""
    raise NotImplementedError

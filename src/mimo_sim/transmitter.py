"""MIMO-OFDM transmitter."""

import numpy as np


def generate_bits(
    num_bits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate a random binary bit sequence.

    Args:
        num_bits: Number of bits to generate. Must be positive.
        rng: NumPy random generator for reproducible results.

    Returns:
        One-dimensional binary NumPy array with shape
        (num_bits,) and dtype int8, containing zeros and ones.

    Raises:
        ValueError: If num_bits is not positive.
    """
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    return rng.integers(0, 2, size=num_bits, dtype=np.int8)


def modulate(
    bits: np.ndarray,
    modulation_type: str,
) -> np.ndarray:
    """Map binary bits to normalized QPSK or 16QAM symbols.

    QPSK uses two bits per symbol, while 16QAM uses four.
    The constellation is normalized to unit average power.

    Args:
        bits: Nonempty one-dimensional binary array
            containing zeros and ones. Its length must be
            divisible by the bits per symbol.
        modulation_type: Modulation scheme, either
            "QPSK" or "16QAM".

    Returns:
        One-dimensional complex symbol array with shape
        (N / B,), where N is the number of input bits and
        B is 2 for QPSK or 4 for 16QAM.

    Raises:
        ValueError: If modulation is unsupported, bit
            values are not binary, the array is empty,
            or the bit count is incompatible with modulation.
    """
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
    """Perform OFDM IFFT modulation and cyclic prefix insertion.

    Args:
        resource_grid: Frequency-domain complex resource
            grid with shape (S, Nfft, Nt), where S is the
            number of OFDM symbols, Nfft is the FFT size,
            and Nt is the number of transmit antennas.
        cp_length: Cyclic prefix length (Ncp), which must
            satisfy 0 < Ncp <= Nfft.

    Returns:
        Time-domain complex OFDM signal with shape
        (S, Nfft + Ncp, Nt), including the cyclic prefix.

    Raises:
        ValueError: If the resource grid is not
            three-dimensional or the CP length is invalid.
    """
    if resource_grid.ndim != 3:
        raise ValueError("Expected shape (S, Nfft, Nt)")

    fft_size = resource_grid.shape[1]

    if not 0 < cp_length <= fft_size:
        raise ValueError("Invalid CP length")

    time_signal = np.fft.ifft(resource_grid, axis=1)
    cyclic_prefix = time_signal[:, -cp_length:, :]

    return np.concatenate(
        [cyclic_prefix, time_signal],
        axis=1,
    )


def map_to_resource_grid(
    symbols: np.ndarray,
    fft_size: int,
    num_subcarriers: int,
) -> np.ndarray:
    """Map active OFDM subcarriers into a centered FFT grid.

    Active subcarriers are placed in the middle of the
    frequency grid. An IFFT shift is then applied to
    obtain the frequency-bin ordering expected by np.fft.ifft.

    Args:
        symbols: Complex active-subcarrier symbols with
            shape (S, K, Nt), where K is num_subcarriers.
        fft_size: Total number of FFT points (Nfft).
        num_subcarriers: Number of active subcarriers (K),
            satisfying 0 < K <= Nfft.

    Returns:
        Complex FFT resource grid with shape (S, Nfft, Nt).
        Inactive subcarriers contain zeros.

    Raises:
        ValueError: If the input is not three-dimensional,
            the active-subcarrier count does not match K,
            or the subcarrier configuration is invalid.
    """
    if symbols.ndim != 3:
        raise ValueError("Expected shape (S, K, Nt)")

    if symbols.shape[1] != num_subcarriers:
        raise ValueError("Subcarrier count mismatch")

    if not 0 < num_subcarriers <= fft_size:
        raise ValueError("Invalid subcarrier configuration")

    start = (fft_size - num_subcarriers) // 2
    grid = np.zeros(
        (symbols.shape[0], fft_size, symbols.shape[2]),
        dtype=np.complex128,
    )

    grid[:, start : start + num_subcarriers, :] = symbols

    return np.fft.ifftshift(grid, axes=1)

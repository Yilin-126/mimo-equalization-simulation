"""MIMO-OFDM receiver module."""

import numpy as np


def ofdm_demodulate(
    received_signal: np.ndarray,
    cp_length: int,
    fft_size: int,
) -> np.ndarray:
    """Remove the cyclic prefix and perform OFDM FFT demodulation.

    Args:
        received_signal: Received time-domain OFDM signal
            with shape (S, Nfft + Ncp, Nr), where S is the
            number of OFDM symbols and Nr is the number
            of receive antennas.
        cp_length: Cyclic prefix length (Ncp), which must
            be positive and no greater than fft_size.
        fft_size: Number of FFT points (Nfft).

    Returns:
        Frequency-domain complex OFDM signal with shape
        (S, Nfft, Nr), obtained by removing the cyclic
        prefix and applying FFT along the sample axis.

    Raises:
        ValueError: If the input is not three-dimensional,
            the CP length is invalid, or the OFDM symbol
            length does not match Nfft + Ncp.
    """
    if received_signal.ndim != 3:
        raise ValueError("Expected shape (S, samples, Nr)")

    if not 0 < cp_length <= fft_size:
        raise ValueError("Invalid CP length")

    if received_signal.shape[1] != fft_size + cp_length:
        raise ValueError("Invalid OFDM symbol length")

    time_signal = received_signal[:, cp_length:, :]

    return np.fft.fft(time_signal, axis=1)


def demodulate(
    symbols: np.ndarray,
    modulation_type: str,
) -> np.ndarray:
    """Perform hard-decision QPSK or 16QAM symbol demodulation.

    QPSK decisions use the signs of the real and imaginary
    components. 16QAM decisions additionally use the
    normalized decision threshold 2 / sqrt(10).

    Args:
        symbols: Nonempty one-dimensional complex symbol
            array with shape (N,).
        modulation_type: Modulation scheme, either
            "QPSK" or "16QAM".

    Returns:
        Flattened binary bit array of dtype int8.
        QPSK produces shape (2 * N,), while 16QAM
        produces shape (4 * N,).

    Raises:
        ValueError: If symbols are empty or not
            one-dimensional, or modulation is unsupported.
        TypeError: If the input symbols are not complex.
    """
    symbols = np.asarray(symbols)

    if symbols.ndim != 1 or symbols.size == 0:
        raise ValueError("Expected non-empty 1D symbols")

    if not np.iscomplexobj(symbols):
        raise TypeError("Symbols must be complex")

    if modulation_type == "QPSK":
        bits = np.column_stack((symbols.real < 0, symbols.imag < 0))

    elif modulation_type == "16QAM":
        threshold = 2 / np.sqrt(10)
        bits = np.column_stack(
            (
                symbols.real < 0,
                np.abs(symbols.real) < threshold,
                symbols.imag < 0,
                np.abs(symbols.imag) < threshold,
            )
        )

    else:
        raise ValueError("Unsupported modulation")

    return bits.astype(np.int8).reshape(-1)

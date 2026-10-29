"""MIMO-OFDM接收机模块。"""

import numpy as np


def ofdm_demodulate(
    received_signal: np.ndarray,
    cp_length: int,
    fft_size: int,
) -> np.ndarray:
    """Remove cyclic prefix and perform FFT."""
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
    """Hard-decision QPSK and 16QAM demodulation."""
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

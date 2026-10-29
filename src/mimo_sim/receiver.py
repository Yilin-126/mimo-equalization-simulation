"""MIMO-OFDM接收机模块。"""

import numpy as np


def ofdm_demodulate(
    received_signal: np.ndarray,
    cp_length: int,
    fft_size: int,
) -> np.ndarray:
    """去除循环前缀并完成FFT解调。"""
    raise NotImplementedError


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

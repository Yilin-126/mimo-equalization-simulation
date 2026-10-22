"""MIMO-OFDM接收机模块。"""

import numpy as np


def ofdm_demodulate(
    received_signal: np.ndarray,
    cp_length: int,
    fft_size: int,
) -> np.ndarray:
    """去除循环前缀并完成FFT解调。"""
    raise NotImplementedError


def demodulate(
    symbols: np.ndarray,
    modulation_type: str,
) -> np.ndarray:
    """QPSK/16QAM硬判决解调。"""
    raise NotImplementedError

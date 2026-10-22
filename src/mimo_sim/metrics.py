"""MIMO-OFDM链路性能指标计算模块。"""

import numpy as np


def calculate_ber(
    transmitted_bits: np.ndarray,
    received_bits: np.ndarray,
) -> float:
    """计算误码率BER。"""
    raise NotImplementedError


def calculate_evm(
    reference_symbols: np.ndarray,
    estimated_symbols: np.ndarray,
) -> float:
    """计算均方根误差向量幅度EVM。"""
    raise NotImplementedError


def calculate_mse(
    true_channel: np.ndarray,
    estimated_channel: np.ndarray,
) -> float:
    """计算信道估计均方误差MSE。"""
    raise NotImplementedError

"""MIMO信道模型。"""

import numpy as np


def add_awgn(
    signal: np.ndarray,
    noise_variance: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """添加复高斯白噪声。"""
    raise NotImplementedError


def generate_mimo_channel(
    num_tx: int,
    num_rx: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成MIMO信道矩阵，形状为(Nr, Nt)。"""
    raise NotImplementedError

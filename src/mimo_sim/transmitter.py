"""MIMO-OFDM发射机模块。

定义随机比特生成、数字调制和OFDM调制接口。
"""

import numpy as np


def generate_bits(
    num_bits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """生成随机二进制比特序列。

    Args:
        num_bits: 需要生成的比特数量。
        rng: NumPy随机数生成器。

    Returns:
        一维二进制比特数组。
    """
    raise NotImplementedError("Bit generation is not implemented yet.")


def modulate(
    bits: np.ndarray,
    modulation_type: str,
) -> np.ndarray:
    """数字调制接口，支持QPSK和16QAM。

    Args:
        bits: 输入二进制比特数组。
        modulation_type: 调制方式。

    Returns:
        调制后的复数符号数组。
    """
    raise NotImplementedError("Modulation is not implemented yet.")


def ofdm_modulate(
    resource_grid: np.ndarray,
    cp_length: int,
) -> np.ndarray:
    """OFDM调制接口。

    Args:
        resource_grid: 频域资源网格。
        cp_length: 循环前缀采样点数。

    Returns:
        添加CP后的时域复数信号。
    """
    raise NotImplementedError("OFDM modulation is not implemented yet.")

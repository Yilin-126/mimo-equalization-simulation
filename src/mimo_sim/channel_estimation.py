"""DMRS信道估计模块。"""

import numpy as np


def ls_channel_estimate(
    received_pilots: np.ndarray,
    reference_pilots: np.ndarray,
) -> np.ndarray:
    """基于已知导频进行LS信道估计。"""
    raise NotImplementedError


def interpolate_channel(
    pilot_estimates: np.ndarray,
    pilot_indices: np.ndarray,
    num_subcarriers: int,
) -> np.ndarray:
    """将导频位置的信道估计插值到全部子载波。"""
    raise NotImplementedError

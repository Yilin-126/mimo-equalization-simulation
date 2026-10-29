"""MIMO-OFDM performance metrics."""

import numpy as np


def calculate_ber(
    transmitted_bits: np.ndarray,
    received_bits: np.ndarray,
) -> float:
    """Calculate bit error rate."""
    tx = np.asarray(transmitted_bits)
    rx = np.asarray(received_bits)

    if tx.shape != rx.shape or tx.size == 0:
        raise ValueError("Invalid bit dimensions")

    if not np.all((tx == 0) | (tx == 1)):
        raise ValueError("Invalid transmitted bits")

    if not np.all((rx == 0) | (rx == 1)):
        raise ValueError("Invalid received bits")

    return float(np.mean(tx != rx))


def calculate_evm(
    reference_symbols: np.ndarray,
    estimated_symbols: np.ndarray,
) -> float:
    """Calculate normalized RMS EVM."""
    reference = np.asarray(reference_symbols)
    estimated = np.asarray(estimated_symbols)

    if reference.shape != estimated.shape or reference.size == 0:
        raise ValueError("Invalid symbol dimensions")

    reference_power = np.sum(np.abs(reference) ** 2)

    if reference_power <= 0:
        raise ValueError("Reference power must be positive")

    error_power = np.sum(np.abs(estimated - reference) ** 2)

    return float(np.sqrt(error_power / reference_power))


def calculate_mse(
    true_channel: np.ndarray,
    estimated_channel: np.ndarray,
) -> float:
    """Calculate channel estimation mean squared error."""
    true = np.asarray(true_channel)
    estimated = np.asarray(estimated_channel)

    if true.shape != estimated.shape or true.size == 0:
        raise ValueError("Invalid channel dimensions")

    return float(np.mean(np.abs(estimated - true) ** 2))

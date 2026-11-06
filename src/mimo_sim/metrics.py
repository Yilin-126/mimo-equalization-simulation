"""MIMO-OFDM performance metrics."""

import numpy as np


def calculate_ber(
    transmitted_bits: np.ndarray,
    received_bits: np.ndarray,
) -> float:
    """Calculate the bit error rate (BER).

    Args:
        transmitted_bits: Original binary bit array.
        received_bits: Recovered binary bit array with
            the same shape as transmitted_bits.

    Returns:
        Bit error rate as a float between 0 and 1.

    Raises:
        ValueError: If arrays are empty, shapes differ,
            or either array contains non-binary values.
    """
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
    """Calculate normalized root mean square EVM.

    Args:
        reference_symbols: Ideal reference constellation
            symbols stored as a complex NumPy array.
        estimated_symbols: Received or equalized complex
            symbols with the same shape as reference_symbols.

    Returns:
        Normalized RMS EVM as a nonnegative float.
        The result is a linear ratio, not a percentage or dB.

    Raises:
        ValueError: If arrays are empty, shapes differ,
            or reference symbol power is zero.
    """
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
    """Calculate channel estimation mean squared error.

    Args:
        true_channel: Reference complex channel coefficients
            or channel matrices.
        estimated_channel: Estimated complex channel values
            with the same shape as true_channel.

    Returns:
        Mean squared magnitude of the channel estimation
        error as a nonnegative float.

    Raises:
        ValueError: If arrays are empty or their shapes differ.
    """
    true = np.asarray(true_channel)
    estimated = np.asarray(estimated_channel)

    if true.shape != estimated.shape or true.size == 0:
        raise ValueError("Invalid channel dimensions")

    return float(np.mean(np.abs(estimated - true) ** 2))

"""Performance metric tests."""

import numpy as np
import pytest

from src.mimo_sim.metrics import (
    calculate_ber,
    calculate_evm,
    calculate_mse,
)


def test_ber() -> None:
    """Verify BER with fixed bit sequences."""
    tx = np.array([0, 1, 0, 1])
    rx = np.array([0, 0, 0, 1])

    assert calculate_ber(tx, rx) == 0.25


def test_zero_ber() -> None:
    """Verify error-free transmission."""
    bits = np.array([0, 1, 1, 0])

    assert calculate_ber(bits, bits) == 0.0


def test_evm() -> None:
    """Verify normalized RMS EVM."""
    reference = np.array([1 + 0j, 1 + 0j])
    estimated = np.array([1.1 + 0j, 0.9 + 0j])

    assert calculate_evm(reference, estimated) == pytest.approx(0.1)


def test_zero_evm() -> None:
    """Verify ideal constellation EVM."""
    symbols = np.array([1 + 1j, -1 - 1j])

    assert calculate_evm(symbols, symbols) == 0.0


def test_channel_mse() -> None:
    """Verify channel estimation MSE."""
    true = np.array([1 + 0j, 2 + 0j])
    estimated = np.array([1.1 + 0j, 1.9 + 0j])

    assert calculate_mse(true, estimated) == pytest.approx(0.01)


def test_invalid_ber_dimensions() -> None:
    """Reject mismatched bit arrays."""
    with pytest.raises(ValueError):
        calculate_ber(np.array([0, 1]), np.array([0]))

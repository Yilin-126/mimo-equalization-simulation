"""Tests for digital modulation."""

import numpy as np
import pytest

from src.mimo_sim.transmitter import generate_bits, modulate


def test_generate_bits() -> None:
    """Check deterministic bit generation."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    np.testing.assert_array_equal(
        generate_bits(100, rng1),
        generate_bits(100, rng2),
    )


def test_qpsk_mapping() -> None:
    """Check QPSK constellation values."""
    bits = np.array([0, 0, 0, 1, 1, 1, 1, 0])

    expected = np.array([1 + 1j, 1 - 1j, -1 - 1j, -1 + 1j])
    expected = expected / np.sqrt(2)

    np.testing.assert_allclose(
        modulate(bits, "QPSK"),
        expected,
    )


def test_16qam_mapping() -> None:
    """Check 16QAM constellation values."""
    bits = np.array(
        [
            0,
            0,
            0,
            0,
            0,
            1,
            0,
            1,
            1,
            1,
            1,
            1,
            1,
            0,
            1,
            0,
        ]
    )

    expected = np.array([3 + 3j, 1 + 1j, -1 - 1j, -3 - 3j])
    expected = expected / np.sqrt(10)

    np.testing.assert_allclose(
        modulate(bits, "16QAM"),
        expected,
    )


def test_invalid_bit_length() -> None:
    """Reject incomplete symbols."""
    with pytest.raises(ValueError):
        modulate(np.array([0, 1, 0]), "QPSK")


def test_output_shape() -> None:
    """Check symbol count."""
    bits = np.zeros(100, dtype=np.int8)

    assert modulate(bits, "QPSK").shape == (50,)
    assert modulate(bits, "16QAM").shape == (25,)

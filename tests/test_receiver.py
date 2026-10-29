"""Receiver demodulation tests."""

import numpy as np
import pytest

from src.mimo_sim.receiver import demodulate
from src.mimo_sim.transmitter import modulate


@pytest.mark.parametrize("modulation", ["QPSK", "16QAM"])
def test_modulation_roundtrip(modulation: str) -> None:
    """Verify noiseless bit recovery."""
    rng = np.random.default_rng(42)
    bits_per_symbol = 2 if modulation == "QPSK" else 4

    bits = rng.integers(
        0, 2, size=1000 * bits_per_symbol, dtype=np.int8
    )

    symbols = modulate(bits, modulation)
    recovered = demodulate(symbols, modulation)

    np.testing.assert_array_equal(recovered, bits)


def test_demodulation_output_dtype() -> None:
    """Verify recovered bit data type."""
    symbols = np.array([1 + 1j, -1 - 1j]) / np.sqrt(2)

    result = demodulate(symbols, "QPSK")

    assert result.shape == (4,)
    assert result.dtype == np.int8


def test_invalid_modulation() -> None:
    """Reject unsupported modulation types."""
    symbols = np.array([1 + 1j])

    with pytest.raises(ValueError):
        demodulate(symbols, "64QAM")
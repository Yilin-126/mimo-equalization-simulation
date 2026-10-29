"""OFDM modulation and demodulation tests."""

import numpy as np
import pytest

from src.mimo_sim.receiver import ofdm_demodulate
from src.mimo_sim.transmitter import ofdm_modulate


@pytest.mark.parametrize("num_antennas", [1, 2])
def test_ofdm_roundtrip(num_antennas: int) -> None:
    """Verify noiseless OFDM recovery."""
    rng = np.random.default_rng(42)

    grid = rng.standard_normal((14, 2048, num_antennas)) + 1j * rng.standard_normal(
        (14, 2048, num_antennas)
    )

    tx_signal = ofdm_modulate(grid, cp_length=144)
    rx_grid = ofdm_demodulate(
        tx_signal,
        cp_length=144,
        fft_size=2048,
    )

    assert tx_signal.shape == (14, 2192, num_antennas)
    assert rx_grid.shape == grid.shape

    np.testing.assert_allclose(
        rx_grid,
        grid,
        rtol=0,
        atol=1e-10,
    )


def test_invalid_cp_length() -> None:
    """Reject invalid CP length."""
    grid = np.zeros((14, 2048, 1), dtype=np.complex128)

    with pytest.raises(ValueError):
        ofdm_modulate(grid, cp_length=0)


from src.mimo_sim.transmitter import map_to_resource_grid


def test_active_subcarrier_roundtrip() -> None:
    """Verify active subcarriers after OFDM transmission."""
    rng = np.random.default_rng(42)

    symbols = rng.standard_normal((14, 1200, 2)) + 1j * rng.standard_normal(
        (14, 1200, 2)
    )

    tx_grid = map_to_resource_grid(
        symbols,
        fft_size=2048,
        num_subcarriers=1200,
    )

    tx_signal = ofdm_modulate(tx_grid, cp_length=144)

    rx_grid = ofdm_demodulate(
        tx_signal,
        cp_length=144,
        fft_size=2048,
    )

    centered_grid = np.fft.fftshift(rx_grid, axes=1)
    start = (2048 - 1200) // 2

    recovered = centered_grid[:, start : start + 1200, :]

    assert recovered.shape == symbols.shape

    np.testing.assert_allclose(
        recovered,
        symbols,
        rtol=0,
        atol=1e-10,
    )

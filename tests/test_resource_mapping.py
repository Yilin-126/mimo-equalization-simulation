"""Resource mapping tests."""

import numpy as np

from src.mimo_sim.transmitter import map_to_resource_grid


def test_resource_mapping() -> None:
    """Verify active subcarrier placement."""
    symbols = np.ones((14, 1200, 2), dtype=np.complex128)

    grid = map_to_resource_grid(symbols, 2048, 1200)
    centered_grid = np.fft.fftshift(grid, axes=1)

    start = (2048 - 1200) // 2

    assert grid.shape == (14, 2048, 2)

    np.testing.assert_array_equal(
        centered_grid[:, start : start + 1200, :],
        symbols,
    )

    assert np.count_nonzero(centered_grid[:, :start, :]) == 0
    assert np.count_nonzero(centered_grid[:, start + 1200 :, :]) == 0


def test_invalid_mapping() -> None:
    """Reject invalid active subcarrier count."""
    symbols = np.ones((14, 1200, 1), dtype=np.complex128)

    try:
        map_to_resource_grid(symbols, 1024, 1200)
    except ValueError:
        return

    raise AssertionError("Expected ValueError")

"""LS channel estimation tests."""

import numpy as np
import pytest

from src.mimo_sim.channel_estimation import (
    interpolate_channel,
    ls_channel_estimate,
)


def test_ls_estimation() -> None:
    """Verify noiseless LS estimation."""
    pilots = np.array([1 + 0j, -1 + 0j, 1j, -1j])
    true_channel = np.array([0.8 + 0.2j, 0.5 - 0.3j, 1 + 0.1j, 0.7 - 0.2j])

    received = true_channel * pilots
    estimated = ls_channel_estimate(received, pilots)

    np.testing.assert_allclose(
        estimated,
        true_channel,
        atol=1e-12,
    )


def test_ls_output_shape() -> None:
    """Verify channel estimate dimensions."""
    pilots = np.ones((14, 1200), dtype=np.complex128)
    received = 0.5 * pilots

    estimated = ls_channel_estimate(received, pilots)

    assert estimated.shape == (14, 1200)


def test_invalid_pilots() -> None:
    """Reject zero-valued reference pilots."""
    received = np.ones(4, dtype=np.complex128)
    pilots = np.array([1, 1, 0, 1])

    with pytest.raises(ValueError):
        ls_channel_estimate(received, pilots)


def test_channel_interpolation() -> None:
    """Verify complex linear interpolation."""
    pilot_indices = np.array([0, 4, 8])
    pilot_estimates = np.array([1 + 1j, 3 + 3j, 5 + 5j])

    result = interpolate_channel(
        pilot_estimates,
        pilot_indices,
        num_subcarriers=9,
    )

    expected = np.array(
        [
            1 + 1j,
            1.5 + 1.5j,
            2 + 2j,
            2.5 + 2.5j,
            3 + 3j,
            3.5 + 3.5j,
            4 + 4j,
            4.5 + 4.5j,
            5 + 5j,
        ]
    )

    np.testing.assert_allclose(result, expected)


def test_interpolation_shape() -> None:
    """Verify output subcarrier dimension."""
    pilots = np.array([1 + 0j, 2 + 0j])
    indices = np.array([0, 1199])

    result = interpolate_channel(pilots, indices, 1200)

    assert result.shape == (1200,)
    assert result.dtype == np.complex128


def test_invalid_pilot_indices() -> None:
    """Reject unsorted pilot positions."""
    pilots = np.array([1 + 0j, 2 + 0j])
    indices = np.array([5, 2])

    with pytest.raises(ValueError):
        interpolate_channel(pilots, indices, 1200)

"""Tests for MIMO pilot-based LS channel estimation."""

import numpy as np
import pytest

from src.mimo_sim.csi_estimation import estimate_mimo_channel_ls


def test_noiseless_channel_recovery() -> None:
    """Verify exact recovery with noiseless orthogonal pilots."""
    rng = np.random.default_rng(42)

    h = (
        rng.standard_normal((16, 2, 2)) + 1j * rng.standard_normal((16, 2, 2))
    ) / np.sqrt(2)

    noise = np.zeros_like(h)

    estimate = estimate_mimo_channel_ls(
        h,
        noise,
        time_noise_variance=0.0,
        fft_size=32,
    )

    np.testing.assert_allclose(estimate, h, atol=1e-12)


def test_known_ls_estimation_error() -> None:
    """Verify the LS error equals the injected pilot noise."""
    h = np.ones((8, 2, 2), dtype=np.complex128)

    innovations = np.full(
        (8, 2, 2),
        1 + 1j,
        dtype=np.complex128,
    ) / np.sqrt(2)

    variance = 0.0025
    fft_size = 64
    amplitude = 2.0

    estimate = estimate_mimo_channel_ls(
        h,
        innovations,
        variance,
        fft_size,
        amplitude,
    )

    expected_error = np.sqrt(fft_size * variance) * innovations / amplitude

    np.testing.assert_allclose(
        estimate - h,
        expected_error,
        atol=1e-12,
    )


def test_channel_error_decreases_with_pilot_power() -> None:
    """Verify increasing pilot amplitude reduces estimation error."""
    h = np.ones((32, 2, 2), dtype=np.complex128)

    innovations = np.full(
        h.shape,
        0.5 + 0.5j,
        dtype=np.complex128,
    )

    low = estimate_mimo_channel_ls(h, innovations, 0.01, 64, pilot_amplitude=1.0)

    high = estimate_mimo_channel_ls(h, innovations, 0.01, 64, pilot_amplitude=2.0)

    mse_low = np.mean(np.abs(low - h) ** 2)
    mse_high = np.mean(np.abs(high - h) ** 2)

    assert mse_high < mse_low
    np.testing.assert_allclose(
        mse_low / mse_high,
        4.0,
        rtol=1e-12,
    )


def test_invalid_pilot_dimensions() -> None:
    """Reject mismatched channel and pilot noise dimensions."""
    h = np.ones((8, 2, 2), dtype=np.complex128)
    noise = np.ones((7, 2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        estimate_mimo_channel_ls(h, noise, 0.01, 64)


def test_invalid_noise_variance() -> None:
    """Reject negative noise variance."""
    h = np.ones((8, 2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        estimate_mimo_channel_ls(h, np.zeros_like(h), -1.0, 64)

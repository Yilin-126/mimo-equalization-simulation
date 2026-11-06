"""Tests for discrete frequency-selective MIMO channels."""

import numpy as np
import pytest

from src.mimo_sim.multipath import (
    apply_multipath_channel,
    create_multipath_taps,
    multipath_frequency_response,
)


def test_tap_shapes_and_normalization() -> None:
    """Verify tap placement and expected power normalization."""
    base = np.ones((3, 2, 2), dtype=np.complex128)

    taps = create_multipath_taps(
        base,
        delays=[0, 2, 5],
        powers_db=[0.0, -3.0, -6.0],
    )

    assert taps.shape == (6, 2, 2)
    assert np.all(taps[1] == 0)
    assert np.all(taps[3] == 0)
    assert np.all(taps[4] == 0)

    np.testing.assert_allclose(
        np.sum(np.abs(taps[:, 0, 0]) ** 2),
        1.0,
        atol=1e-12,
    )


def test_single_tap_matches_flat_mimo() -> None:
    """Verify zero-delay convolution equals a flat MIMO channel."""
    rng = np.random.default_rng(42)

    tx = rng.standard_normal((2, 40, 2)) + 1j * rng.standard_normal((2, 40, 2))

    h = np.array([[1 + 0.2j, 0.3], [0.2j, 0.9 - 0.1j]])

    actual = apply_multipath_channel(
        tx,
        h[np.newaxis, :, :],
    )

    expected = np.einsum("rt,slt->slr", h, tx)

    np.testing.assert_allclose(actual, expected, atol=1e-12)


def test_ofdm_frequency_domain_equivalence() -> None:
    """Verify multipath convolution agrees with per-bin MIMO H."""
    rng = np.random.default_rng(123)

    nfft = 32
    cp = 8
    num_symbols = 3

    grid = rng.standard_normal((num_symbols, nfft, 2)) + 1j * rng.standard_normal(
        (num_symbols, nfft, 2)
    )

    time_signal = np.fft.ifft(grid, axis=1)

    tx = np.concatenate(
        [time_signal[:, -cp:, :], time_signal],
        axis=1,
    )

    innovations = (
        rng.standard_normal((3, 2, 2)) + 1j * rng.standard_normal((3, 2, 2))
    ) / np.sqrt(2)

    taps = create_multipath_taps(
        innovations,
        delays=[0, 2, 5],
        powers_db=[0, -3, -6],
    )

    rx = apply_multipath_channel(tx, taps)
    actual = np.fft.fft(rx[:, cp:, :], axis=1)

    response = multipath_frequency_response(taps, nfft)

    expected = np.einsum(
        "krt,skt->skr",
        response,
        grid,
    )

    np.testing.assert_allclose(
        actual,
        expected,
        rtol=1e-11,
        atol=1e-11,
    )


def test_invalid_duplicate_delays() -> None:
    """Reject repeated delay locations."""
    base = np.ones((2, 2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        create_multipath_taps(base, [0, 0], [0, -3])


def test_invalid_frequency_response_size() -> None:
    """Reject an FFT smaller than the channel impulse response."""
    taps = np.ones((6, 2, 2), dtype=np.complex128)

    with pytest.raises(ValueError):
        multipath_frequency_response(taps, 4)

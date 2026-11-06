"""Tests for controlled comparisons across MIMO channel scenarios."""

import numpy as np

from src.mimo_sim.random_control import create_frame_samples
from src.mimo_sim.transmitter import (
    map_to_resource_grid,
    modulate,
    ofdm_modulate,
)

CONFIG = {
    "random_control": {
        "enabled": True,
        "bit_seed": 20261106,
        "noise_seed": 20261107,
        "channel_seed": 20261108,
        "reuse_realizations": True,
    }
}

FFT_SIZE = 32
CP_LENGTH = 8
SUBCARRIERS = 16
SYMBOLS = 4
ANTENNAS = 2
BITS_PER_FRAME = SYMBOLS * SUBCARRIERS * ANTENNAS * 2


def make_frame(snr_index: int, frame_index: int) -> dict:
    """Generate a reproducible MIMO-OFDM test frame.

    Args:
        snr_index: Index of the SNR point.
        frame_index: Index of the OFDM frame.

    Returns:
        Shared bits, noise innovations, and channel innovations.
    """
    return create_frame_samples(
        config=CONFIG,
        snr_index=snr_index,
        frame_index=frame_index,
        num_bits=BITS_PER_FRAME,
        noise_shape=(SYMBOLS, FFT_SIZE + CP_LENGTH, ANTENNAS),
        channel_shape=(ANTENNAS, ANTENNAS),
    )


def transmit(bits: np.ndarray) -> np.ndarray:
    """Convert shared bits into a time-domain OFDM signal.

    Args:
        bits: Binary QPSK input sequence.

    Returns:
        Complex OFDM signal with shape
        (SYMBOLS, FFT_SIZE + CP_LENGTH, ANTENNAS).
    """
    symbols = modulate(bits, "QPSK").reshape(SYMBOLS, SUBCARRIERS, ANTENNAS)

    grid = map_to_resource_grid(symbols, FFT_SIZE, SUBCARRIERS)

    return ofdm_modulate(grid, CP_LENGTH)


def test_same_inputs_across_channel_scenarios() -> None:
    """Verify both channel scenarios use identical base samples."""
    samples_a = make_frame(5, 3)
    samples_b = make_frame(5, 3)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(samples_a[key], samples_b[key])


def test_channel_is_the_only_changed_variable() -> None:
    """Verify a changed channel does not change bits or noise."""
    samples = make_frame(5, 3)
    tx = transmit(samples["bits"])

    h_a = np.array(
        [[1.0, 0.3], [0.2, 0.9]],
        dtype=np.complex128,
    )

    h_b = np.array(
        [[0.8, 0.1], [0.4, 1.1]],
        dtype=np.complex128,
    )

    clean_a = np.einsum("rt,slt->slr", h_a, tx)
    clean_b = np.einsum("rt,slt->slr", h_b, tx)

    # Fixed absolute noise variance for both scenarios.
    noise_variance = 0.01
    common_noise = np.sqrt(noise_variance) * samples["noise"]

    received_a = clean_a + common_noise
    received_b = clean_b + common_noise

    np.testing.assert_allclose(
        received_a - clean_a,
        received_b - clean_b,
        rtol=1e-12,
        atol=1e-12,
    )

    np.testing.assert_allclose(
        received_b - received_a,
        clean_b - clean_a,
        rtol=1e-12,
        atol=1e-12,
    )


def test_channel_innovations_can_be_reused() -> None:
    """Verify different models can share base channel innovations."""
    samples = make_frame(5, 3)
    base_channel = samples["channel"]

    identity_channel = base_channel.copy()

    correlation = np.array(
        [[1.0, 0.0], [0.5, np.sqrt(0.75)]],
        dtype=np.float64,
    )

    correlated_channel = correlation @ base_channel

    assert np.array_equal(identity_channel, base_channel)

    np.testing.assert_allclose(
        correlated_channel,
        correlation @ identity_channel,
    )


def test_frame_order_does_not_change_samples() -> None:
    """Verify sample identity does not depend on scenario order."""
    first = make_frame(5, 3)

    make_frame(2, 20)
    make_frame(0, 0)
    make_frame(9, 1)

    repeated = make_frame(5, 3)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(first[key], repeated[key])

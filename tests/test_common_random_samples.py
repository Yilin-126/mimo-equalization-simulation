"""Tests for reproducible common random samples."""

import numpy as np
import pytest

from src.mimo_sim.random_control import create_frame_samples

CONFIG = {
    "random_control": {
        "enabled": True,
        "bit_seed": 20261106,
        "noise_seed": 20261107,
        "channel_seed": 20261108,
        "reuse_realizations": True,
    }
}


def get_samples(snr_index: int, frame_index: int) -> dict:
    """Generate one deterministic test frame.

    Args:
        snr_index: SNR point index.
        frame_index: Frame index.

    Returns:
        Dictionary of common bit, noise, and channel samples.
    """
    return create_frame_samples(
        CONFIG,
        snr_index=snr_index,
        frame_index=frame_index,
        num_bits=1000,
        noise_shape=(14, 128, 2),
        channel_shape=(2, 2),
    )


def test_same_frame_reproducible() -> None:
    """Verify identical frame identifiers reproduce all samples.

    Args:
        None.

    Returns:
        None. Assertions verify reproducibility.
    """
    first = get_samples(5, 3)
    second = get_samples(5, 3)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(first[key], second[key])


def test_frame_order_independent() -> None:
    """Verify frame results do not depend on execution order.

    Args:
        None.

    Returns:
        None. Assertions verify identical frame samples.
    """
    original = get_samples(5, 3)

    get_samples(0, 0)
    get_samples(10, 20)

    repeated = get_samples(5, 3)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(original[key], repeated[key])


def test_different_frames() -> None:
    """Verify distinct frame indices produce distinct samples.

    Args:
        None.

    Returns:
        None. Assertions verify different bit sequences.
    """
    first = get_samples(5, 3)
    second = get_samples(5, 4)

    assert not np.array_equal(first["bits"], second["bits"])


def test_identical_samples_across_scenarios() -> None:
    """Verify scenario labels do not change common random data.

    Args:
        None.

    Returns:
        None. Assertions verify identical base samples.
    """
    zf_mmse_samples = get_samples(5, 3)
    channel_comparison_samples = get_samples(5, 3)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(
            zf_mmse_samples[key],
            channel_comparison_samples[key],
        )


def test_invalid_frame_index() -> None:
    """Reject negative frame indices.

    Args:
        None.

    Returns:
        None. Expects ValueError for invalid input.
    """
    with pytest.raises(ValueError):
        get_samples(5, -1)

"""Tests for common random samples across SNR points."""

import copy

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
        "reuse_across_snr": True,
    }
}


def generate(config: dict, snr_index: int) -> dict:
    """Generate deterministic samples for one SNR point.

    Args:
        config: Random-control configuration.
        snr_index: Index of the requested SNR point.

    Returns:
        Dictionary containing bits, noise, and channel arrays.
    """
    return create_frame_samples(
        config=config,
        snr_index=snr_index,
        frame_index=3,
        num_bits=1000,
        noise_shape=(14, 128, 2),
        channel_shape=(2, 2),
        scenario_id="independent",
    )


def test_reuse_across_snr_enabled() -> None:
    """Check that different SNR points share the same base samples.

    Args:
        None.

    Returns:
        None. Assertions verify sample equality.
    """
    first = generate(CONFIG, 0)
    second = generate(CONFIG, 10)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(first[key], second[key])


def test_reuse_across_snr_disabled() -> None:
    """Check that disabling reuse separates SNR random streams.

    Args:
        None.

    Returns:
        None. Assertions verify distinct samples.
    """
    config = copy.deepcopy(CONFIG)
    config["random_control"]["reuse_across_snr"] = False

    first = generate(config, 0)
    second = generate(config, 10)

    for key in ("bits", "noise", "channel"):
        assert not np.array_equal(first[key], second[key])


def test_disabled_mode_is_reproducible() -> None:
    """Check repeated SNR samples remain deterministic.

    Args:
        None.

    Returns:
        None. Assertions verify reproducibility.
    """
    config = copy.deepcopy(CONFIG)
    config["random_control"]["reuse_across_snr"] = False

    first = generate(config, 5)
    second = generate(config, 5)

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(first[key], second[key])


def test_invalid_reuse_across_snr() -> None:
    """Reject non-boolean cross-SNR reuse settings.

    Args:
        None.

    Returns:
        None. Expects ValueError.
    """
    config = copy.deepcopy(CONFIG)
    config["random_control"]["reuse_across_snr"] = "yes"

    with pytest.raises(ValueError):
        generate(config, 0)

"""Tests for scenario-dependent random realization reuse."""

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
    }
}


def generate(config: dict, scenario: str) -> dict:
    """Generate deterministic samples for one scenario.

    Args:
        config: Random control configuration.
        scenario: Scenario identifier.

    Returns:
        Dictionary of bit, noise, and channel samples.
    """
    return create_frame_samples(
        config=config,
        snr_index=5,
        frame_index=3,
        num_bits=1000,
        noise_shape=(14, 128, 2),
        channel_shape=(2, 2),
        scenario_id=scenario,
    )


def test_reuse_enabled() -> None:
    """Verify different scenarios share identical samples."""
    first = generate(CONFIG, "flat_channel")
    second = generate(CONFIG, "correlated_channel")

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(first[key], second[key])


def test_reuse_disabled() -> None:
    """Verify different scenarios receive different samples."""
    config = copy.deepcopy(CONFIG)
    config["random_control"]["reuse_realizations"] = False

    first = generate(config, "flat_channel")
    second = generate(config, "correlated_channel")

    for key in ("bits", "noise", "channel"):
        assert not np.array_equal(first[key], second[key])


def test_disabled_mode_is_reproducible() -> None:
    """Verify repeated scenario calls remain deterministic."""
    config = copy.deepcopy(CONFIG)
    config["random_control"]["reuse_realizations"] = False

    first = generate(config, "correlated_channel")
    second = generate(config, "correlated_channel")

    for key in ("bits", "noise", "channel"):
        assert np.array_equal(first[key], second[key])


def test_invalid_scenario_identifier() -> None:
    """Reject an empty scenario identifier."""
    with pytest.raises(ValueError):
        generate(CONFIG, "")

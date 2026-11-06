"""Tests for controlled independent and correlated channels."""

import numpy as np
import pytest

from src.mimo_sim.channel_scenarios import (
    apply_channel_model,
    create_scenario_frame,
)

CONFIG = {
    "random_control": {
        "enabled": True,
        "reuse_realizations": True,
        "bit_seed": 20261106,
        "noise_seed": 20261107,
        "channel_seed": 20261108,
    }
}


def generate(scenario_id: str, model: str) -> dict:
    """Generate one controlled test scenario.

    Args:
        scenario_id: Scenario identifier.
        model: Channel model name.

    Returns:
        Dictionary of common inputs and channel outputs.
    """
    return create_scenario_frame(
        config=CONFIG,
        snr_index=5,
        frame_index=3,
        scenario_id=scenario_id,
        model=model,
        num_bits=1000,
        noise_shape=(4, 40, 2),
        num_rx=2,
        num_tx=2,
        correlation=0.7,
    )


def test_same_base_samples_across_models() -> None:
    """Verify different channel models share common inputs."""
    first = generate("flat", "independent")
    second = generate("correlated", "rx_correlated")

    for key in ("bits", "noise", "base_channel"):
        assert np.array_equal(first[key], second[key])


def test_channel_model_changes_only_channel() -> None:
    """Verify the correlated model transforms the channel."""
    first = generate("flat", "independent")
    second = generate("correlated", "rx_correlated")

    assert not np.array_equal(first["channel"], second["channel"])

    factor = np.array(
        [
            [1.0, 0.0],
            [0.7, np.sqrt(1.0 - 0.7**2)],
        ]
    )

    np.testing.assert_allclose(
        second["channel"],
        factor @ first["channel"],
    )


def test_scenario_reproducibility() -> None:
    """Verify that rerunning a scenario reproduces its output."""
    first = generate("correlated", "rx_correlated")
    second = generate("correlated", "rx_correlated")

    for key in ("bits", "noise", "base_channel", "channel"):
        assert np.array_equal(first[key], second[key])


def test_invalid_correlation() -> None:
    """Reject correlation coefficients outside the model range."""
    with pytest.raises(ValueError):
        apply_channel_model(
            np.eye(2, dtype=np.complex128),
            "rx_correlated",
            1.5,
        )

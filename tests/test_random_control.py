"""Tests for independent simulation random generators."""

import numpy as np

from src.mimo_sim.random_control import create_rngs

CONFIG = {
    "random_control": {
        "bit_seed": 20261106,
        "noise_seed": 20261107,
        "channel_seed": 20261108,
    }
}


def test_reproducible_streams() -> None:
    """Verify that identical seeds reproduce identical sequences."""
    first = create_rngs(CONFIG)
    second = create_rngs(CONFIG)

    for name in ("bits", "noise", "channel"):
        assert np.array_equal(
            first[name].random(100),
            second[name].random(100),
        )


def test_independent_stream_consumption() -> None:
    """Verify that using one stream does not advance another."""
    first = create_rngs(CONFIG)
    second = create_rngs(CONFIG)

    first["bits"].random(1000)

    assert np.array_equal(
        first["noise"].random(100),
        second["noise"].random(100),
    )

    assert np.array_equal(
        first["channel"].random(100),
        second["channel"].random(100),
    )


def test_different_seeds() -> None:
    """Verify that distinct seeds generate different sequences."""
    streams = create_rngs(CONFIG)

    bits = streams["bits"].random(100)
    noise = streams["noise"].random(100)
    channel = streams["channel"].random(100)

    assert not np.array_equal(bits, noise)
    assert not np.array_equal(bits, channel)
    assert not np.array_equal(noise, channel)

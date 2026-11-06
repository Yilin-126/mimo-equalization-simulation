"""Tests for adaptive BER stopping rules."""

import pytest

from src.mimo_sim.ber_stopping import get_stop_reason


def test_target_reached() -> None:
    """Stop when both equalizers reach target errors."""
    assert get_stop_reason(250, 210, 1000, 200, 5000) == "target_reached"


def test_one_target_not_reached() -> None:
    """Continue when only one equalizer reaches target."""
    assert get_stop_reason(250, 180, 1000, 200, 5000) is None


def test_max_bits_reached() -> None:
    """Stop when maximum bit count is reached."""
    assert get_stop_reason(19, 13, 2016000, 200, 2016000) == "max_bits_reached"


def test_invalid_parameters() -> None:
    """Reject invalid stopping parameters."""
    with pytest.raises(ValueError):
        get_stop_reason(0, 0, 100, 0, 1000)

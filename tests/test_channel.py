"""AWGN channel tests."""

import numpy as np
import pytest
from src.mimo_sim.channel import (
    add_awgn,
    generate_frequency_channel,
    generate_mimo_channel,
)


def test_awgn_shape() -> None:
    """Verify output shape and dtype."""
    signal = np.ones((14, 2048, 2), dtype=np.complex128)
    rng = np.random.default_rng(42)

    output = add_awgn(signal, 0.1, rng)

    assert output.shape == signal.shape
    assert output.dtype == np.complex128


def test_awgn_snr() -> None:
    """Verify measured SNR against target."""
    rng = np.random.default_rng(42)

    signal = np.ones(100000, dtype=np.complex128)
    target_snr_db = 10.0

    signal_power = np.mean(np.abs(signal) ** 2)
    noise_variance = signal_power / (10 ** (target_snr_db / 10))

    received = add_awgn(signal, noise_variance, rng)
    measured_noise = received - signal

    measured_snr_db = 10 * np.log10(signal_power / np.mean(np.abs(measured_noise) ** 2))

    assert abs(measured_snr_db - target_snr_db) < 0.2


def test_zero_noise() -> None:
    """Verify zero-noise behavior."""
    signal = np.ones(100, dtype=np.complex128)
    rng = np.random.default_rng(42)

    np.testing.assert_array_equal(
        add_awgn(signal, 0.0, rng),
        signal,
    )


def test_invalid_noise_variance() -> None:
    """Reject negative noise power."""
    signal = np.ones(10, dtype=np.complex128)
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        add_awgn(signal, -0.1, rng)


from src.mimo_sim.channel import generate_mimo_channel


@pytest.mark.parametrize("num_antennas", [1, 2])
def test_mimo_channel_shape(num_antennas: int) -> None:
    """Verify channel matrix dimensions."""
    rng = np.random.default_rng(42)
    channel = generate_mimo_channel(num_antennas, num_antennas, rng)

    assert channel.shape == (num_antennas, num_antennas)
    assert channel.dtype == np.complex128


def test_mimo_channel_reproducibility() -> None:
    """Verify deterministic channel generation."""
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)

    np.testing.assert_array_equal(
        generate_mimo_channel(2, 2, rng1),
        generate_mimo_channel(2, 2, rng2),
    )


def test_mimo_channel_power() -> None:
    """Verify average channel coefficient power."""
    rng = np.random.default_rng(42)
    samples = np.array([generate_mimo_channel(2, 2, rng) for _ in range(2000)])

    average_power = np.mean(np.abs(samples) ** 2)

    assert abs(average_power - 1.0) < 0.05


def test_invalid_antenna_count() -> None:
    """Reject invalid antenna configuration."""
    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        generate_mimo_channel(0, 2, rng)


def test_frequency_channel_shape() -> None:
    """Verify frequency-domain MIMO channel dimensions."""
    rng = np.random.default_rng(42)

    channel = generate_frequency_channel(14, 1200, 2, 2, rng)

    assert channel.shape == (14, 1200, 2, 2)
    assert channel.dtype == np.complex128

    np.testing.assert_array_equal(
        channel[0, 0],
        channel[-1, -1],
    )


def test_mimo_matrix_multiplication() -> None:
    """Verify Y = HX and array dimension ordering."""
    channel = np.array([[1 + 0j, 0.5 + 0j], [0 + 0j, 1 + 0j]])
    transmitted = np.array([1 + 1j, 1 - 1j])

    received = channel @ transmitted

    expected = np.array([1.5 + 0.5j, 1 - 1j])

    np.testing.assert_allclose(received, expected)
    assert received.shape == (2,)

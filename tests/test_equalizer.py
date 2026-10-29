"""MIMO equalizer unit tests."""

import numpy as np
import pytest

from src.mimo_sim.equalizer import mmse_equalize, zf_equalize


@pytest.fixture
def mimo_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create a fixed 2x2 MIMO test case."""
    channel = np.array([[1 + 0.5j, 0.2 + 0.1j], [0.3 - 0.2j, 1 - 0.4j]])
    transmitted = np.array([1 + 1j, -1 + 1j])
    received = channel @ transmitted
    return channel, transmitted, received


def test_zf_recovery(mimo_example) -> None:
    """Verify noiseless ZF recovery."""
    channel, transmitted, received = mimo_example

    result = zf_equalize(received, channel)

    np.testing.assert_allclose(result, transmitted, atol=1e-12)


def test_mmse_zero_noise(mimo_example) -> None:
    """Verify MMSE equals ZF at zero noise."""
    channel, transmitted, received = mimo_example

    result = mmse_equalize(received, channel, 0.0)

    np.testing.assert_allclose(result, transmitted, atol=1e-12)


def test_mmse_regularization(mimo_example) -> None:
    """Verify MMSE against an independently formed solution."""
    channel, _, received = mimo_example
    noise_variance = 0.1

    expected = (
        np.linalg.inv(channel.conj().T @ channel + noise_variance * np.eye(2))
        @ channel.conj().T
        @ received
    )

    result = mmse_equalize(received, channel, noise_variance)

    np.testing.assert_allclose(result, expected, atol=1e-12)


def test_output_shape(mimo_example) -> None:
    """Verify equalized signal dimensions."""
    channel, _, received = mimo_example

    assert zf_equalize(received, channel).shape == (2,)
    assert mmse_equalize(received, channel, 0.1).shape == (2,)


def test_invalid_dimensions() -> None:
    """Reject mismatched receive dimensions."""
    channel = np.eye(2, dtype=np.complex128)
    received = np.ones(3, dtype=np.complex128)

    with pytest.raises(ValueError):
        zf_equalize(received, channel)


def test_singular_channel() -> None:
    """Reject rank-deficient channels in ZF."""
    channel = np.array([[1, 1], [2, 2]], dtype=np.complex128)
    received = np.ones(2, dtype=np.complex128)

    with pytest.raises(ValueError):
        zf_equalize(received, channel)


def test_negative_noise_variance() -> None:
    """Reject invalid noise variance."""
    channel = np.eye(2, dtype=np.complex128)
    received = np.ones(2, dtype=np.complex128)

    with pytest.raises(ValueError):
        mmse_equalize(received, channel, -0.1)


def test_mmse_low_snr_performance() -> None:
    """Compare ZF and MMSE mean square errors at low SNR."""
    rng = np.random.default_rng(42)

    channel = np.array(
        [[1.0, 0.95], [0.95, 1.0]],
        dtype=np.complex128,
    )

    noise_variance = 0.1
    num_trials = 1000

    zf_error = 0.0
    mmse_error = 0.0

    for _ in range(num_trials):
        transmitted = (
            rng.choice([-1, 1], size=2) + 1j * rng.choice([-1, 1], size=2)
        ) / np.sqrt(2)

        noise = np.sqrt(noise_variance / 2) * (
            rng.standard_normal(2) + 1j * rng.standard_normal(2)
        )

        received = channel @ transmitted + noise

        zf_result = zf_equalize(received, channel)
        mmse_result = mmse_equalize(
            received,
            channel,
            noise_variance,
        )

        zf_error += np.mean(np.abs(zf_result - transmitted) ** 2)
        mmse_error += np.mean(np.abs(mmse_result - transmitted) ** 2)

    zf_error /= num_trials
    mmse_error /= num_trials

    assert mmse_error < zf_error

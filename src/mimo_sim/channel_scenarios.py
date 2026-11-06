"""Controlled MIMO channel scenarios for reproducible experiments."""

import numpy as np

from src.mimo_sim.random_control import create_frame_samples


def apply_channel_model(
    base_channel: np.ndarray,
    model: str,
    correlation: float = 0.0,
) -> np.ndarray:
    """Generate flat Rayleigh MIMO channels from common innovations.

    Args:
        base_channel: Unit-variance complex Gaussian channel
            innovations with shape (Nr, Nt).
        model: Either "independent" or "rx_correlated".
        correlation: Real receive-correlation coefficient
            between 0 and 1, used for the 2x2 model.

    Returns:
        Complex MIMO channel matrix with shape (Nr, Nt).

    Raises:
        ValueError: If the channel shape, model, or
            correlation coefficient is unsupported.
    """
    h = np.asarray(base_channel, dtype=np.complex128)

    if h.ndim != 2 or min(h.shape) <= 0:
        raise ValueError("Invalid MIMO channel dimensions")

    if model == "independent":
        return h.copy()

    if model != "rx_correlated":
        raise ValueError("Unsupported channel model")

    if h.shape[0] != 2:
        raise ValueError("Receive correlation requires Nr=2")

    if not np.isfinite(correlation) or not 0 <= correlation < 1:
        raise ValueError("Correlation must satisfy 0 <= rho < 1")

    # Receive-side correlation factor: R = L @ L^H.
    factor = np.array(
        [
            [1.0, 0.0],
            [correlation, np.sqrt(1.0 - correlation**2)],
        ],
        dtype=np.float64,
    )

    return factor @ h


def create_scenario_frame(
    config: dict,
    snr_index: int,
    frame_index: int,
    scenario_id: str,
    model: str,
    num_bits: int,
    noise_shape: tuple[int, ...],
    num_rx: int,
    num_tx: int,
    correlation: float = 0.0,
) -> dict[str, np.ndarray]:
    """Create controlled input samples for one MIMO scenario.

    Args:
        config: Simulation configuration with random seeds
            and reuse_realizations settings.
        snr_index: Index of the SNR point.
        frame_index: Index of the simulation frame.
        scenario_id: Unique name of the channel scenario.
        model: Channel model name.
        num_bits: Number of transmitted bits.
        noise_shape: Required complex noise array shape.
        num_rx: Number of receive antennas.
        num_tx: Number of transmit antennas.
        correlation: Receive-side correlation coefficient.

    Returns:
        Dictionary containing:
            bits: Transmitted binary bit array.
            noise: Shared unit-variance noise innovations.
            base_channel: Original complex Gaussian channel.
            channel: Channel matrix after model transformation.

    Raises:
        ValueError: If dimensions or model settings are invalid.
    """
    if type(num_rx) is not int or type(num_tx) is not int or num_rx <= 0 or num_tx <= 0:
        raise ValueError("Invalid antenna configuration")

    samples = create_frame_samples(
        config=config,
        snr_index=snr_index,
        frame_index=frame_index,
        num_bits=num_bits,
        noise_shape=noise_shape,
        channel_shape=(num_rx, num_tx),
        scenario_id=scenario_id,
    )

    channel = apply_channel_model(
        samples["channel"],
        model,
        correlation,
    )

    return {
        "bits": samples["bits"],
        "noise": samples["noise"],
        "base_channel": samples["channel"],
        "channel": channel,
    }

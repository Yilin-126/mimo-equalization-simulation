"""Pilot-based LS channel estimation for MIMO-OFDM."""

import numpy as np


def estimate_mimo_channel_ls(
    true_channel: np.ndarray,
    pilot_noise: np.ndarray,
    time_noise_variance: float,
    fft_size: int,
    pilot_amplitude: float = 1.0,
) -> np.ndarray:
    """Estimate a per-subcarrier MIMO channel using orthogonal pilots.

    Each transmit antenna uses an orthogonal training slot.
    With an identity pilot matrix, the received pilot array is:

        Y_pilot = pilot_amplitude * H + W

    Complex AWGN is specified in the OFDM time domain.
    For NumPy's unnormalized FFT, the corresponding frequency-
    domain noise variance is fft_size * time_noise_variance.

    Args:
        true_channel: True frequency-domain MIMO channel,
            shape (K, Nr, Nt).
        pilot_noise: Independent unit-variance complex Gaussian
            innovations, shape (K, Nr, Nt).
        time_noise_variance: Nonnegative complex AWGN variance
            per receive antenna and time-domain sample.
        fft_size: Positive OFDM FFT size.
        pilot_amplitude: Positive real training-symbol amplitude.

    Returns:
        LS estimated complex channel with shape (K, Nr, Nt).

    Raises:
        ValueError: If dimensions, pilot settings, or noise
            variance are invalid.
        TypeError: If channel or noise inputs are not complex.
    """
    channel = np.asarray(true_channel)
    innovations = np.asarray(pilot_noise)

    if not np.iscomplexobj(channel) or not np.iscomplexobj(innovations):
        raise TypeError("Channel and pilot noise must be complex")

    if (
        channel.ndim != 3
        or min(channel.shape) <= 0
        or channel.shape != innovations.shape
    ):
        raise ValueError("Invalid MIMO pilot dimensions")

    if (
        type(fft_size) is not int
        or fft_size <= 0
        or not np.isfinite(time_noise_variance)
        or time_noise_variance < 0
        or not np.isfinite(pilot_amplitude)
        or pilot_amplitude <= 0
    ):
        raise ValueError("Invalid pilot or noise parameters")

    frequency_noise_variance = fft_size * time_noise_variance

    received_pilots = (
        pilot_amplitude * channel + np.sqrt(frequency_noise_variance) * innovations
    )

    return received_pilots / pilot_amplitude

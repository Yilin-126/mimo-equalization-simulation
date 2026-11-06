"""Frequency-selective tapped-delay MIMO channel model."""

import numpy as np


def create_multipath_taps(
    base_channel: np.ndarray,
    delays: list[int],
    powers_db: list[float],
) -> np.ndarray:
    """Build a normalized discrete tapped-delay MIMO channel.

    Args:
        base_channel: Independent unit-variance complex Gaussian
            innovations, shape (L, Nr, Nt).
        delays: Distinct nonnegative integer tap delays in samples.
        powers_db: Relative tap powers in dB, one per delay.

    Returns:
        Complex channel impulse response with shape
        (max_delay + 1, Nr, Nt). Unused delays contain zeros.
        The total expected tap power per antenna pair is one.

    Raises:
        ValueError: If delay, power, or dimension settings are invalid.
        TypeError: If the base innovations are not complex.
    """
    innovations = np.asarray(base_channel)

    if innovations.ndim != 3 or not np.iscomplexobj(innovations):
        raise TypeError("Expected complex channel innovations (L, Nr, Nt)")

    if (
        len(delays) == 0
        or len(delays) != len(powers_db)
        or innovations.shape[0] != len(delays)
        or min(innovations.shape) <= 0
    ):
        raise ValueError("Invalid multipath dimensions")

    if any(type(delay) is not int or delay < 0 for delay in delays):
        raise ValueError("Tap delays must be nonnegative integers")

    if len(set(delays)) != len(delays):
        raise ValueError("Tap delays must be distinct")

    powers = np.asarray(powers_db, dtype=np.float64)

    if not np.all(np.isfinite(powers)):
        raise ValueError("Invalid tap powers")

    # Scale relative powers safely before normalization.
    linear = 10 ** ((powers - np.max(powers)) / 10)
    normalized_power = linear / np.sum(linear)

    taps = np.zeros(
        (max(delays) + 1, *innovations.shape[1:]),
        dtype=np.complex128,
    )

    for index, delay in enumerate(delays):
        taps[delay] = innovations[index] * np.sqrt(normalized_power[index])

    return taps


def apply_multipath_channel(
    transmitted: np.ndarray,
    taps: np.ndarray,
) -> np.ndarray:
    """Apply causal MIMO multipath convolution to OFDM blocks.

    Each OFDM block is processed separately. This is valid for
    the useful FFT interval when CP length covers the maximum
    tap delay. No noise is added by this function.

    Args:
        transmitted: Complex CP-prefixed signal with shape
            (S, Nfft + CP, Nt).
        taps: Channel impulse response with shape
            (D + 1, Nr, Nt).

    Returns:
        Received complex signal with shape
        (S, Nfft + CP, Nr).

    Raises:
        ValueError: If dimensions or antenna counts are invalid.
        TypeError: If input arrays are not complex.
    """
    tx = np.asarray(transmitted)
    h = np.asarray(taps)

    if not np.iscomplexobj(tx) or not np.iscomplexobj(h):
        raise TypeError("Transmitted signals and taps must be complex")

    if tx.ndim != 3 or h.ndim != 3:
        raise ValueError("Invalid MIMO multipath dimensions")

    symbols, block_length, num_tx = tx.shape
    tap_count, num_rx, channel_tx = h.shape

    if (
        symbols <= 0
        or block_length <= 0
        or tap_count <= 0
        or num_rx <= 0
        or num_tx != channel_tx
    ):
        raise ValueError("Invalid signal or channel configuration")

    if tap_count > block_length:
        raise ValueError("Channel delay exceeds the input block")

    received = np.zeros(
        (symbols, block_length, num_rx),
        dtype=np.complex128,
    )

    for delay in range(tap_count):
        received[:, delay:, :] += np.einsum(
            "rt,slt->slr",
            h[delay],
            tx[:, : block_length - delay, :],
        )

    return received


def multipath_frequency_response(
    taps: np.ndarray,
    fft_size: int,
) -> np.ndarray:
    """Calculate the MIMO channel frequency response.

    Args:
        taps: Complex channel impulse response with shape
            (D + 1, Nr, Nt).
        fft_size: Positive FFT size, at least the tap count.

    Returns:
        Channel frequency response with shape
        (Nfft, Nr, Nt), in NumPy FFT bin order.

    Raises:
        ValueError: If dimensions or FFT size are invalid.
        TypeError: If channel taps are not complex.
    """
    h = np.asarray(taps)

    if not np.iscomplexobj(h):
        raise TypeError("Channel taps must be complex")

    if (
        h.ndim != 3
        or min(h.shape) <= 0
        or type(fft_size) is not int
        or fft_size < h.shape[0]
    ):
        raise ValueError("Invalid frequency-response dimensions")

    return np.fft.fft(h, n=fft_size, axis=0)

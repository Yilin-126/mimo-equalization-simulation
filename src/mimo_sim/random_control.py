"""Reproducible random control for MIMO-OFDM experiments."""

import numpy as np


def create_rngs(config: dict) -> dict[str, np.random.Generator]:
    """Create independent random generators for simulation.

    Args:
        config: Configuration dictionary containing
            random_control and its three seed values.

    Returns:
        Dictionary containing generators named
        bits, noise, and channel.

    Raises:
        KeyError: If required seed fields are missing.
        TypeError: If seed values are invalid types.
        ValueError: If seed values are outside valid ranges.
    """
    seeds = config["random_control"]

    return {
        "bits": np.random.default_rng(seeds["bit_seed"]),
        "noise": np.random.default_rng(seeds["noise_seed"]),
        "channel": np.random.default_rng(seeds["channel_seed"]),
    }


def create_frame_samples(
    config: dict,
    snr_index: int,
    frame_index: int,
    num_bits: int,
    noise_shape: tuple[int, ...],
    channel_shape: tuple[int, ...],
) -> dict[str, np.ndarray]:
    """Generate reproducible common random samples for one frame.

    Each random stream is initialized from its configured seed,
    SNR index, and frame index. Thus, generating one frame does
    not advance or change the random data of another frame.

    The resulting noise and channel arrays contain unit-variance
    circularly symmetric complex Gaussian samples.

    Args:
        config: Configuration containing random_control seeds.
        snr_index: Nonnegative index of the SNR test point.
        frame_index: Nonnegative index of the simulation frame.
        num_bits: Positive number of transmitted bits.
        noise_shape: Required shape of the complex noise array.
        channel_shape: Required shape of the complex channel
            innovation array.

    Returns:
        Dictionary with three independent arrays:
            bits: Binary int8 array with shape (num_bits,).
            noise: Unit-variance complex Gaussian noise array.
            channel: Unit-variance complex Gaussian channel
                innovation array.

        Identical inputs generate identical arrays regardless
        of which other frames were generated previously.

    Raises:
        ValueError: If indices, dimensions, or the enabled
            configuration are invalid.
    """
    control = config["random_control"]

    if control.get("enabled") is not True:
        raise ValueError("Common random control must be enabled")

    if (
        not isinstance(snr_index, int)
        or not isinstance(frame_index, int)
        or not isinstance(num_bits, int)
        or min(snr_index, frame_index) < 0
        or num_bits <= 0
    ):
        raise ValueError("Invalid frame sampling parameters")

    for shape in (noise_shape, channel_shape):
        if (
            not isinstance(shape, tuple)
            or not shape
            or any(not isinstance(size, int) or size <= 0 for size in shape)
        ):
            raise ValueError("Invalid sample shape")

    def make_rng(seed: int) -> np.random.Generator:
        """Initialize a deterministic generator for this frame.

        Args:
            seed: Nonnegative base seed from the configuration.

        Returns:
            NumPy generator indexed by the SNR and frame.

        Raises:
            ValueError: If the base seed is invalid.
        """
        if not isinstance(seed, int) or seed < 0:
            raise ValueError("Invalid random seed")

        sequence = np.random.SeedSequence([seed, snr_index, frame_index])
        return np.random.default_rng(sequence)

    bits_rng = make_rng(control["bit_seed"])
    noise_rng = make_rng(control["noise_seed"])
    channel_rng = make_rng(control["channel_seed"])

    bits = bits_rng.integers(
        0,
        2,
        size=num_bits,
        dtype=np.int8,
    )

    def complex_gaussian(
        rng: np.random.Generator,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Generate unit-variance complex Gaussian samples.

        Args:
            rng: NumPy random generator.
            shape: Output array dimensions.

        Returns:
            Complex array with independent real and
            imaginary standard-normal components scaled
            by 1/sqrt(2).
        """
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)

        return (real + 1j * imag) / np.sqrt(2)

    noise = complex_gaussian(noise_rng, noise_shape)
    channel = complex_gaussian(channel_rng, channel_shape)

    return {
        "bits": bits,
        "noise": noise,
        "channel": channel,
    }

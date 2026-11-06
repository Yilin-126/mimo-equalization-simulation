"""Reproducible random control for MIMO-OFDM experiments."""

import hashlib

import numpy as np


def create_rngs(config: dict) -> dict[str, np.random.Generator]:
    """Create independent random generators for simulation.

    Args:
        config: Configuration containing random_control
            and three independent random seeds.

    Returns:
        Dictionary containing generators named
        bits, noise, and channel.

    Raises:
        KeyError: If required seed fields are missing.
        TypeError: If seed values have invalid types.
        ValueError: If seed values are invalid.
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
    scenario_id: str = "default",
) -> dict[str, np.ndarray]:
    """Generate deterministic random samples for a simulation frame.

    Independent random streams are used for transmitted bits,
    complex noise, and channel innovations.

    When reuse_realizations is True, scenario_id is ignored,
    so different scenarios share the same random samples.

    When reuse_realizations is False, scenario_id is included
    in the seed sequence. Different scenario identifiers
    therefore produce independent reproducible streams.

    Args:
        config: Simulation configuration containing
            random_control settings and seeds.
        snr_index: Nonnegative SNR point index.
        frame_index: Nonnegative frame index.
        num_bits: Positive number of transmitted bits.
        noise_shape: Shape of the complex noise array.
        channel_shape: Shape of the complex channel array.
        scenario_id: Nonempty identifier for the experiment
            scenario. Defaults to "default" for compatibility.

    Returns:
        Dictionary containing:
            bits: Binary int8 array of shape (num_bits,).
            noise: Unit-variance complex Gaussian array
                with shape noise_shape.
            channel: Unit-variance complex Gaussian array
                with shape channel_shape.

    Raises:
        KeyError: If required configuration values are missing.
        TypeError: If configuration or seed types are invalid.
        ValueError: If indices, shapes, flags, or scenario
            identifier are invalid.
    """
    control = config["random_control"]

    if control.get("enabled") is not True:
        raise ValueError("Random control must be enabled")

    reuse = control.get("reuse_realizations", True)
    reuse_across_snr = control.get("reuse_across_snr", False)

    if not isinstance(reuse, bool):
        raise ValueError("reuse_realizations must be boolean")

    if not isinstance(reuse_across_snr, bool):
        raise ValueError("reuse_across_snr must be boolean")

    effective_snr_index = 0 if reuse_across_snr else snr_index

    if (
        type(snr_index) is not int
        or type(frame_index) is not int
        or type(num_bits) is not int
        or min(snr_index, frame_index) < 0
        or num_bits <= 0
    ):
        raise ValueError("Invalid frame sampling parameters")

    if not isinstance(scenario_id, str) or not scenario_id:
        raise ValueError("scenario_id must be a nonempty string")

    for shape in (noise_shape, channel_shape):
        if (
            not isinstance(shape, tuple)
            or not shape
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise ValueError("Invalid sample shape")

    scenario_words = []

    if not reuse:
        digest = hashlib.sha256(scenario_id.encode("utf-8")).digest()

        scenario_words = [
            int.from_bytes(digest[0:4], "little"),
            int.from_bytes(digest[4:8], "little"),
        ]

    def make_rng(seed: int) -> np.random.Generator:
        """Create a deterministic random generator.

        Args:
            seed: Nonnegative base seed.

        Returns:
            Generator indexed by SNR, frame, and optionally
            the scenario identifier.

        Raises:
            ValueError: If the seed is invalid.
        """
        if type(seed) is not int or seed < 0:
            raise ValueError("Invalid random seed")

        sequence = np.random.SeedSequence(
            [
                seed,
                effective_snr_index,
                frame_index,
                *scenario_words,
            ]
        )

        return np.random.default_rng(sequence)

    def complex_gaussian(
        rng: np.random.Generator,
        shape: tuple[int, ...],
    ) -> np.ndarray:
        """Generate unit-variance circular complex Gaussian data.

        Args:
            rng: NumPy random generator.
            shape: Output array shape.

        Returns:
            Complex NumPy array with unit average power.
        """
        real = rng.standard_normal(shape)
        imag = rng.standard_normal(shape)

        return (real + 1j * imag) / np.sqrt(2)

    bits_rng = make_rng(control["bit_seed"])
    noise_rng = make_rng(control["noise_seed"])
    channel_rng = make_rng(control["channel_seed"])

    bits = bits_rng.integers(
        0,
        2,
        size=num_bits,
        dtype=np.int8,
    )

    noise = complex_gaussian(
        noise_rng,
        noise_shape,
    )

    channel = complex_gaussian(
        channel_rng,
        channel_shape,
    )

    return {
        "bits": bits,
        "noise": noise,
        "channel": channel,
    }

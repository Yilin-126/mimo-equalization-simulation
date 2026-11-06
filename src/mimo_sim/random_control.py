"""Manage independent random streams for reproducible simulations."""

import numpy as np


def create_rngs(config: dict) -> dict[str, np.random.Generator]:
    """Create independent random generators for simulation components.

    Separate random streams prevent changes in one component
    from consuming random numbers assigned to another component.

    Args:
        config: Simulation configuration dictionary containing
            the random_control section with bit_seed,
            noise_seed, and channel_seed.

    Returns:
        Dictionary containing three NumPy generators:
            bits: Transmitted-bit generation.
            noise: Additive noise generation.
            channel: Random channel generation.

    Raises:
        KeyError: If required configuration entries are missing.
        ValueError: If any seed is invalid.
        TypeError: If the configuration values have invalid types.
    """
    seeds = config["random_control"]

    return {
        "bits": np.random.default_rng(seeds["bit_seed"]),
        "noise": np.random.default_rng(seeds["noise_seed"]),
        "channel": np.random.default_rng(seeds["channel_seed"]),
    }

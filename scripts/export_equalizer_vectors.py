"""Export fixed Python ZF/MMSE reference vectors for C validation."""

import json
from pathlib import Path

import numpy as np

from src.mimo_sim.equalizer import mmse_equalize, zf_equalize

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "tests" / "fixtures" / "equalizer_reference.json"


def encode_complex(values: np.ndarray) -> list:
    """Convert complex arrays to JSON-compatible real/imag pairs.

    Args:
        values: Complex NumPy array of arbitrary shape.

    Returns:
        Nested list with [real, imag] as the last dimension.
    """
    array = np.asarray(values, dtype=np.complex128)
    return np.stack((array.real, array.imag), axis=-1).tolist()


def make_reference(
    name: str,
    channel: np.ndarray,
    transmitted: np.ndarray,
    snr_db: float,
    seed: int,
) -> dict:
    """Generate one deterministic MIMO equalization test vector.

    Args:
        name: Unique test case identifier.
        channel: Complex channel matrix with shape (Nr, Nt).
        transmitted: Transmitted symbols with shape (Nt,).
        snr_db: SNR referenced to unit transmit symbol power.
        seed: Random seed for complex Gaussian noise.

    Returns:
        Dictionary containing inputs, outputs, and metadata.
    """
    rng = np.random.default_rng(seed)
    nr, nt = channel.shape

    noise_variance = 10.0 ** (-snr_db / 10.0)

    noise = np.sqrt(noise_variance / 2.0) * (
        rng.standard_normal(nr) + 1j * rng.standard_normal(nr)
    )

    received = channel @ transmitted + noise

    zf_result = zf_equalize(received, channel)
    mmse_result = mmse_equalize(
        received,
        channel,
        noise_variance,
    )

    return {
        "name": name,
        "num_rx": nr,
        "num_tx": nt,
        "snr_db": snr_db,
        "noise_variance": noise_variance,
        "condition_number": float(np.linalg.cond(channel)),
        "channel": encode_complex(channel),
        "transmitted": encode_complex(transmitted),
        "noise": encode_complex(noise),
        "received": encode_complex(received),
        "zf_expected": encode_complex(zf_result),
        "mmse_expected": encode_complex(mmse_result),
    }


def main() -> None:
    """Generate and save the complete equalizer reference dataset.

    Args:
        None.

    Returns:
        None. Writes the reference JSON file to tests/fixtures.
    """
    q = 1.0 / np.sqrt(2.0)

    cases = [
        (
            "siso_normal",
            np.array([[0.8 + 0.3j]], dtype=np.complex128),
            np.array([q + 1j * q]),
        ),
        (
            "siso_weak",
            np.array([[0.05 + 0.02j]], dtype=np.complex128),
            np.array([q - 1j * q]),
        ),
        (
            "mimo_normal",
            np.array(
                [
                    [1.0 + 0.2j, 0.3 - 0.1j],
                    [0.2 + 0.1j, 0.9 - 0.2j],
                ],
                dtype=np.complex128,
            ),
            np.array([q + 1j * q, -q + 1j * q]),
        ),
        (
            "mimo_near_singular",
            np.array(
                [
                    [1.0 + 0j, 1.0 + 0j],
                    [1.0 + 0j, 1.001 + 0j],
                ],
                dtype=np.complex128,
            ),
            np.array([q + 1j * q, -q - 1j * q]),
        ),
    ]

    vectors = []

    for index, (name, channel, transmitted) in enumerate(cases):
        for snr_db in (0.0, 20.0):
            vectors.append(
                make_reference(
                    name=f"{name}_{int(snr_db)}db",
                    channel=channel,
                    transmitted=transmitted,
                    snr_db=snr_db,
                    seed=2026 + index * 10 + int(snr_db),
                )
            )

    dataset = {
        "version": 1,
        "description": "Python ZF/MMSE reference for C validation",
        "complex_format": "[real, imag]",
        "matrix_layout": "row-major (Nr, Nt)",
        "noise_model": "CN(0, noise_variance * I)",
        "symbol_power": 1.0,
        "cases": vectors,
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(dataset, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Exported {len(vectors)} reference cases")
    print(f"Saved to {OUTPUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()

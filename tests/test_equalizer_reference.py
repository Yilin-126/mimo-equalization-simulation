"""Regression tests for frozen ZF/MMSE reference vectors."""

import json
from pathlib import Path

import numpy as np
import pytest

from src.mimo_sim.equalizer import mmse_equalize, zf_equalize

REFERENCE_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "equalizer_reference.json"
)


def decode_complex(values: list) -> np.ndarray:
    """Decode stored real/imag pairs into complex arrays.

    Args:
        values: Nested lists with [real, imag] pairs.

    Returns:
        NumPy complex128 array preserving original dimensions.
    """
    data = np.asarray(values, dtype=np.float64)
    return data[..., 0] + 1j * data[..., 1]


def load_reference_cases() -> list:
    """Read fixed reference cases from JSON.

    Args:
        None.

    Returns:
        List of stored ZF/MMSE test cases.
    """
    with REFERENCE_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    return dataset["cases"]


CASES = load_reference_cases()


@pytest.mark.parametrize(
    "case",
    CASES,
    ids=[case["name"] for case in CASES],
)
def test_frozen_equalizer_reference(case: dict) -> None:
    """Compare current Python outputs with frozen reference values.

    Args:
        case: One saved MIMO input/output reference case.

    Returns:
        None. Raises an assertion on numerical mismatch.
    """
    channel = decode_complex(case["channel"])
    received = decode_complex(case["received"])

    expected_zf = decode_complex(case["zf_expected"])
    expected_mmse = decode_complex(case["mmse_expected"])

    assert channel.shape == (
        case["num_rx"],
        case["num_tx"],
    )
    assert received.shape == (case["num_rx"],)

    zf_result = zf_equalize(received, channel)
    mmse_result = mmse_equalize(
        received,
        channel,
        case["noise_variance"],
    )

    np.testing.assert_allclose(
        zf_result,
        expected_zf,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        mmse_result,
        expected_mmse,
        rtol=1e-10,
        atol=1e-10,
    )

"""Plot C/Python numerical validation errors."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.check_c_equalizer import (
    REFERENCE_PATH,
    TOLERANCE,
    decode_complex,
    load_library,
    run_c_equalizer,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Plot absolute C/Python equalizer errors for frozen cases.

    Args:
        None. Frozen references and compiled C library
        are loaded from the existing project files.

    Returns:
        None. Saves a numerical validation figure to
        results/c_python_error_comparison.png.

    Raises:
        FileNotFoundError: If the reference file or DLL is missing.
        RuntimeError: If the C equalizer returns an error.
        AssertionError: If numerical errors exceed tolerance.
    """
    library = load_library()

    with REFERENCE_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    labels = []
    errors = []

    for case in dataset["cases"]:
        channel = decode_complex(case["channel"])
        received = decode_complex(case["received"])
        noise_variance = case["noise_variance"]

        for method, key in (
            ("ZF", "zf_expected"),
            ("MMSE", "mmse_expected"),
        ):
            expected = decode_complex(case[key])
            actual = run_c_equalizer(
                library,
                channel,
                received,
                method,
                noise_variance,
            )

            error = float(np.max(np.abs(actual - expected)))

            if error > TOLERANCE:
                raise AssertionError(f"{case['name']} {method} exceeds tolerance")

            labels.append(f"{case['name']}\n{method}")
            errors.append(error)

    fig, ax = plt.subplots(figsize=(13, 5))

    positions = np.arange(len(labels))
    ax.bar(positions, errors, color="#3975B7")
    ax.axhline(
        TOLERANCE,
        color="#D64545",
        linestyle="--",
        label=f"Tolerance = {TOLERANCE:.0e}",
    )

    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=55, ha="right", fontsize=8)
    ax.set_ylabel("Maximum absolute complex error")
    ax.set_title("C/Python ZF and MMSE Numerical Validation")
    ax.set_yscale("symlog", linthresh=1e-15)
    ax.grid(axis="y", alpha=0.25)
    ax.legend()

    fig.tight_layout()

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    path = output / "c_python_error_comparison.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")
    print(f"Comparisons: {len(errors)}")
    print(f"Maximum absolute error: {max(errors):.3e}")


if __name__ == "__main__":
    main()

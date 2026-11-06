"""Visualize adaptive BER simulation sample counts and stopping reasons."""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "results" / "ber_comparison.csv"


def main() -> None:
    """Plot tested bit counts and stopping reasons by SNR.

    Args:
        None. Reads results/ber_comparison.csv.

    Returns:
        None. Saves results/adaptive_ber_statistics.png.

    Raises:
        FileNotFoundError: If the BER CSV file is missing.
        KeyError: If required CSV columns are absent.
        ValueError: If a stopping reason is unsupported.
    """
    with CSV_PATH.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    snr = np.array([float(row["SNR_dB"]) for row in rows])
    bits = np.array([int(row["Bits"]) for row in rows])
    reasons = [row["Stop_Reason"] for row in rows]

    colors = []
    for reason in reasons:
        if reason == "target_reached":
            colors.append("#3975B7")
        elif reason == "max_bits_reached":
            colors.append("#E49A35")
        else:
            raise ValueError(f"Unknown stopping reason: {reason}")

    fig, ax = plt.subplots(figsize=(10, 5))

    ax.bar(snr, bits / 1e6, width=1.5, color=colors)

    ax.axhline(
        2.016,
        color="#777777",
        linestyle="--",
        linewidth=1.3,
        label="Maximum bit budget",
    )

    ax.set_xticks(snr)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("Tested bits (millions)")
    ax.set_title("Adaptive BER Simulation: Sample Count by SNR")

    from matplotlib.patches import Patch

    ax.legend(
        handles=[
            Patch(color="#3975B7", label="Target errors reached"),
            Patch(color="#E49A35", label="Maximum bit budget reached"),
            plt.Line2D(
                [0],
                [0],
                color="#777777",
                linestyle="--",
                label="Maximum bit budget",
            ),
        ]
    )

    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    fig.tight_layout()

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    path = output / "adaptive_ber_statistics.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)

    print(f"Saved: {path}")


if __name__ == "__main__":
    main()

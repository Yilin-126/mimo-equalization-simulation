"""Benchmark Python and C ZF/MMSE equalizers."""

import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from scripts.check_c_equalizer import load_library, run_c_equalizer
from src.mimo_sim.equalizer import mmse_equalize, zf_equalize
from src.mimo_sim.random_control import create_rngs

ROOT = Path(__file__).resolve().parents[1]
SIZES = (64, 256, 1200)
ANTENNA_COUNTS = (1, 2)
METHODS = ("ZF", "MMSE")
REPEATS = 5
NOISE_VARIANCE = 0.1
TOLERANCE = 1e-8


def make_dataset(config: dict, n: int, count: int) -> tuple:
    """Generate identical inputs for both implementations.

    Args:
        config: Simulation configuration including random seeds
            and the fixed 2x2 channel matrix.
        n: Number of transmit and receive antennas, 1 or 2.
        count: Number of independent received vectors.

    Returns:
        Channel matrix with shape (n, n) and received
        signal array with shape (count, n).
    """
    real = np.asarray(config["performance"]["channel_real"])
    imag = np.asarray(config["performance"]["channel_imag"])
    full_h = real + 1j * imag
    h = np.ascontiguousarray(full_h[:n, :n])

    if np.linalg.matrix_rank(h) < n:
        raise ValueError("Benchmark channel must have full rank")

    rngs = create_rngs(config)
    bits_rng = rngs["bits"]
    noise_rng = rngs["noise"]

    x = (
        bits_rng.standard_normal((count, n)) + 1j * bits_rng.standard_normal((count, n))
    ) / np.sqrt(2)

    noise = np.sqrt(NOISE_VARIANCE / 2) * (
        noise_rng.standard_normal((count, n))
        + 1j * noise_rng.standard_normal((count, n))
    )

    y = x @ h.T + noise
    return h, np.ascontiguousarray(y)


def measure(func, repeats: int = REPEATS) -> float:
    """Measure median elapsed time of repeated function calls.

    Args:
        func: Callable performing one complete benchmark batch.
        repeats: Number of timed repetitions.

    Returns:
        Median elapsed time in milliseconds.
    """
    func()

    durations = []
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        durations.append((time.perf_counter() - start) * 1000)

    return float(np.median(durations))


def main() -> None:
    """Benchmark C via ctypes against Python equalizer functions.

    Args:
        None. Reads the project configuration and C DLL.

    Returns:
        None. Saves CSV timing results and a speedup plot
        in the results directory.

    Raises:
        AssertionError: If C/Python outputs differ beyond
            the accepted numerical tolerance.
        FileNotFoundError: If required inputs are missing.
    """
    with (ROOT / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    library = load_library()
    rows = []

    for n in ANTENNA_COUNTS:
        for count in SIZES:
            h, received = make_dataset(config, n, count)

            for method in METHODS:
                if method == "ZF":

                    def python_one(y):
                        return zf_equalize(y, h)

                else:

                    def python_one(y):
                        return mmse_equalize(y, h, NOISE_VARIANCE)

                def python_batch():
                    return [python_one(y) for y in received]

                def c_batch():
                    return [
                        run_c_equalizer(
                            library,
                            h,
                            y,
                            method,
                            NOISE_VARIANCE,
                        )
                        for y in received
                    ]

                # Validate numerical agreement before timing.
                py_results = python_batch()
                c_results = c_batch()

                max_error = max(
                    float(np.max(np.abs(a - b))) for a, b in zip(py_results, c_results)
                )

                if max_error > TOLERANCE:
                    raise AssertionError(
                        f"{n}x{n} {count} {method}: " f"error={max_error:.3e}"
                    )

                python_ms = measure(python_batch)
                c_ms = measure(c_batch)

                speedup = python_ms / c_ms

                rows.append(
                    (
                        f"{n}x{n}",
                        count,
                        method,
                        python_ms,
                        c_ms,
                        speedup,
                        max_error,
                    )
                )

                print(
                    f"{n}x{n} | K={count:4d} | {method:4s} | "
                    f"Python={python_ms:9.3f} ms | "
                    f"C/ctypes={c_ms:9.3f} ms | "
                    f"Speedup={speedup:.2f}x | "
                    f"Error={max_error:.2e}"
                )

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    csv_path = output / "c_python_benchmark.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Antennas",
                "Subcarriers",
                "Method",
                "Python_ms",
                "C_ctypes_ms",
                "Speedup",
                "Max_Error",
            ]
        )
        writer.writerows(rows)

    labels = [f"{n}x{n}\nK={k}" for n in ANTENNA_COUNTS for k in SIZES]

    x = np.arange(len(labels))
    width = 0.36

    zf_speedups = [row[5] for row in rows if row[2] == "ZF"]
    mmse_speedups = [row[5] for row in rows if row[2] == "MMSE"]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(
        x - width / 2,
        zf_speedups,
        width,
        label="ZF",
    )
    ax.bar(
        x + width / 2,
        mmse_speedups,
        width,
        label="MMSE",
    )
    ax.axhline(
        1.0,
        linestyle="--",
        color="gray",
        label="Equal execution time",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Speedup: Python time / C ctypes time")
    ax.set_title("ZF/MMSE: Python vs C Through ctypes")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.legend()

    fig.tight_layout()

    plot_path = output / "c_python_benchmark.png"
    fig.savefig(plot_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {plot_path}")


if __name__ == "__main__":
    main()

"""Controlled BER comparison across flat Rayleigh MIMO scenarios."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.mimo_sim.channel_scenarios import apply_channel_model
from src.mimo_sim.random_control import create_frame_samples
from src.mimo_sim.receiver import demodulate, ofdm_demodulate
from src.mimo_sim.transmitter import (
    map_to_resource_grid,
    modulate,
    ofdm_modulate,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Compare independent and correlated Rayleigh MIMO channels.

    Both scenarios use the same bits, base channel innovations,
    and complex noise innovations for each SNR/frame pair.

    Noise power is calculated from the independent reference
    channel, then held identical across the compared scenarios.
    Every curve uses the same number of frames at each SNR.

    Args:
        None. Settings are read from configs/simulation.json.

    Returns:
        None. Saves a CSV and a BER figure under results/.

    Raises:
        ValueError: If scenario configuration or sample
            budgets are invalid.
        KeyError: If required configuration fields are missing.
    """
    with (ROOT / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    sim = config["simulation"]
    perf = config["performance"]
    comparison = config["scenario_comparison"]
    random_control = config["random_control"]

    scenarios = comparison["scenarios"]
    names = [item["name"] for item in scenarios]

    if (
        comparison["enabled"] is not True
        or comparison["shared_frame_stopping"] is not True
        or random_control["enabled"] is not True
        or random_control["reuse_realizations"] is not True
    ):
        raise ValueError("Shared comparison control must be enabled")

    if (
        len(scenarios) != 2
        or len(set(names)) != 2
        or comparison["noise_policy"] != "reference_receive_power"
    ):
        raise ValueError("Expected two controlled channel scenarios")

    reference_name = comparison["reference_scenario"]
    reference = next(
        (item for item in scenarios if item["name"] == reference_name),
        None,
    )

    if reference is None or reference["model"] != "independent":
        raise ValueError("Reference scenario must be independent Rayleigh")

    k = int(system["num_subcarriers"])
    nfft = int(system["fft_size"])
    cp = int(system["cp_length"])
    num_symbols = int(system["num_ofdm_symbols"])
    nt = int(config["transmitter"]["num_tx_antennas"])
    nr = int(config["receiver"]["num_rx_antennas"])
    modulation = config["transmitter"]["modulation"]

    if modulation not in ("QPSK", "16QAM"):
        raise ValueError("Unsupported modulation")

    if nt != 2 or nr != 2 or not (0 < k <= nfft):
        raise ValueError("This experiment requires 2x2 MIMO and valid OFDM")

    if cp <= 0 or cp > nfft or num_symbols <= 0:
        raise ValueError("Invalid OFDM settings")

    bits_per_symbol = {"QPSK": 2, "16QAM": 4}[modulation]
    bits_per_frame = num_symbols * k * nt * bits_per_symbol

    target_errors = int(perf["target_errors"])
    max_bits = int(perf["max_bits_per_snr"])
    max_frames = max_bits // bits_per_frame

    min_frames = int(comparison["min_frames_per_snr"])

    if not 1 <= min_frames <= max_frames:
        raise ValueError("Invalid minimum frame count")

    if target_errors <= 0 or max_frames <= 0:
        raise ValueError("Invalid adaptive BER stopping settings")

    snr_values = np.arange(
        sim["snr_start_db"],
        sim["snr_stop_db"] + 1,
        sim["snr_step_db"],
    )

    if sim["snr_step_db"] <= 0 or len(snr_values) == 0:
        raise ValueError("Invalid SNR scan configuration")

    start_subcarrier = (nfft - k) // 2
    identity = np.eye(nt, dtype=np.complex128)
    records = []

    for snr_index, snr_db in enumerate(snr_values):
        counters = {name: {"ZF": 0, "MMSE": 0} for name in names}
        total_bits = 0
        stop_reason = "max_bits_reached"

        for frame_index in range(max_frames):
            samples = create_frame_samples(
                config=config,
                snr_index=snr_index,
                frame_index=frame_index,
                num_bits=bits_per_frame,
                noise_shape=(num_symbols, nfft + cp, nr),
                channel_shape=(nr, nt),
            )

            bits = samples["bits"]
            symbols = modulate(bits, modulation).reshape(num_symbols, k, nt)
            grid = map_to_resource_grid(symbols, nfft, k)
            tx = ofdm_modulate(grid, cp)

            # Transform the same base channel for every scenario.
            channels = {
                item["name"]: apply_channel_model(
                    samples["channel"],
                    item["model"],
                    float(item["correlation"]),
                )
                for item in scenarios
            }

            clean_reference = np.einsum(
                "rt,slt->slr",
                channels[reference_name],
                tx,
            )

            reference_power = float(np.mean(np.abs(clean_reference) ** 2))
            noise_variance = reference_power / (10 ** (float(snr_db) / 10))

            # Exactly the same noise array for both scenarios.
            shared_noise = np.sqrt(noise_variance) * samples["noise"]

            for item in scenarios:
                name = item["name"]
                h = channels[name]

                clean = np.einsum("rt,slt->slr", h, tx)
                rx = clean + shared_noise

                rx_grid = ofdm_demodulate(rx, cp, nfft)
                centered = np.fft.fftshift(rx_grid, axes=1)
                y = centered[:, start_subcarrier : start_subcarrier + k, :]

                wzf = np.linalg.pinv(h)
                hermitian = h.conj().T

                wmmse = np.linalg.solve(
                    hermitian @ h + nfft * noise_variance * identity,
                    hermitian,
                )

                zf_symbols = y @ wzf.T
                mmse_symbols = y @ wmmse.T

                zf_bits = demodulate(zf_symbols.reshape(-1), modulation)
                mmse_bits = demodulate(mmse_symbols.reshape(-1), modulation)

                counters[name]["ZF"] += int(np.count_nonzero(zf_bits != bits))
                counters[name]["MMSE"] += int(np.count_nonzero(mmse_bits != bits))

            total_bits += bits.size

            if frame_index + 1 >= min_frames and all(
                errors >= target_errors
                for method_errors in counters.values()
                for errors in method_errors.values()
            ):
                stop_reason = "target_reached"
                break

        for name in names:
            for method in ("ZF", "MMSE"):
                errors = counters[name][method]
                ber = errors / total_bits
                records.append(
                    (
                        name,
                        float(snr_db),
                        method,
                        ber,
                        total_bits,
                        errors,
                        stop_reason,
                    )
                )

        print(
            f"SNR={snr_db:>3} dB | Bits={total_bits:>8} | "
            + " | ".join(
                f"{name}: "
                f"ZF={counters[name]['ZF'] / total_bits:.6f}, "
                f"MMSE={counters[name]['MMSE'] / total_bits:.6f}"
                for name in names
            )
            + f" | {stop_reason}"
        )

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)

    csv_path = output / "scenario_ber_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Scenario",
                "SNR_dB",
                "Method",
                "BER",
                "Bits",
                "Errors",
                "Stop_Reason",
            ]
        )
        writer.writerows(records)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    styles = {
        "ZF": ("o", "-"),
        "MMSE": ("s", "--"),
    }

    for name in names:
        for method in ("ZF", "MMSE"):
            subset = [row for row in records if row[0] == name and row[2] == method]
            snr = np.array([row[1] for row in subset])
            ber = np.array([row[3] for row in subset])
            marker, line = styles[method]

            # Zero-error observations cannot be plotted on a log axis.
            visible_ber = np.where(ber > 0, ber, np.nan)

            ax.semilogy(
                snr,
                visible_ber,
                marker=marker,
                linestyle=line,
                label=f"{name} / {method}",
            )

    ax.set_xlabel("Reference-channel receive SNR (dB)")
    ax.set_ylabel("Measured bit error rate")
    ax.set_title("2x2 MIMO-OFDM: Controlled Rayleigh Scenario Comparison")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    image_path = output / "scenario_ber_comparison.png"
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {image_path}")


if __name__ == "__main__":
    main()

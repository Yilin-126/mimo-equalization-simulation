"""Adaptive ZF/MMSE BER-SNR comparison for 2x2 MIMO-OFDM."""

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.mimo_sim.ber_stopping import get_stop_reason
from src.mimo_sim.random_control import create_frame_samples
from src.mimo_sim.receiver import demodulate, ofdm_demodulate
from src.mimo_sim.transmitter import (
    map_to_resource_grid,
    modulate,
    ofdm_modulate,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    """Run reproducible adaptive ZF/MMSE BER-SNR simulations.

    Load parameters from the project configuration file and
    generate deterministic common random samples for each
    SNR index and frame index.

    ZF and MMSE share the same transmitted bits, channel
    matrix, and noisy received signal.

    Args:
        None. Parameters are loaded from configs/simulation.json.

    Returns:
        None. Saves the BER data and comparison figure to
        results/ber_comparison.csv and ber_comparison.png.

    Raises:
        FileNotFoundError: If the configuration is missing.
        KeyError: If required configuration entries are missing.
        ValueError: If modulation, channel, or stopping
            parameters are invalid.
    """
    with (ROOT / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    sim = config["simulation"]
    perf = config["performance"]

    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]
    s = system["num_ofdm_symbols"]

    nt = config["transmitter"]["num_tx_antennas"]
    nr = config["receiver"]["num_rx_antennas"]
    modulation = config["transmitter"]["modulation"]

    if modulation not in ("QPSK", "16QAM"):
        raise ValueError("Unsupported modulation")

    bits_per_symbol = {
        "QPSK": 2,
        "16QAM": 4,
    }[modulation]

    h = np.array(perf["channel_real"]) + 1j * np.array(perf["channel_imag"])

    if h.shape != (nr, nt) or np.linalg.matrix_rank(h) < nt:
        raise ValueError("Invalid MIMO channel matrix")

    snr_values = np.arange(
        sim["snr_start_db"],
        sim["snr_stop_db"] + 1,
        sim["snr_step_db"],
    )

    target_errors = int(perf["target_errors"])
    max_bits = int(perf["max_bits_per_snr"])

    bits_per_frame = s * k * nt * bits_per_symbol

    if target_errors <= 0 or max_bits < bits_per_frame:
        raise ValueError("Invalid BER stopping parameters")

    max_frames = max_bits // bits_per_frame

    start = (nfft - k) // 2

    wh = h.conj().T
    wzf = np.linalg.pinv(h)
    identity = np.eye(nt)

    results = []

    for snr_index, snr_db in enumerate(snr_values):
        errors_zf = 0
        errors_mmse = 0
        total_bits = 0

        for frame_index in range(max_frames):
            samples = create_frame_samples(
                config=config,
                snr_index=snr_index,
                frame_index=frame_index,
                num_bits=bits_per_frame,
                noise_shape=(s, nfft + cp, nr),
                channel_shape=(nr, nt),
            )

            bits = samples["bits"]

            symbols = modulate(bits, modulation).reshape(s, k, nt)

            grid = map_to_resource_grid(symbols, nfft, k)

            tx = ofdm_modulate(grid, cp)

            clean = np.einsum("rt,slt->slr", h, tx)

            signal_power = np.mean(np.abs(clean) ** 2)

            noise_var = signal_power / (10 ** (snr_db / 10))

            # Shared complex noise realization for both methods.
            noise = np.sqrt(noise_var) * samples["noise"]

            rx = clean + noise

            rx_grid = ofdm_demodulate(rx, cp, nfft)

            centered = np.fft.fftshift(rx_grid, axes=1)

            y = centered[:, start : start + k, :]

            wmmse = np.linalg.solve(
                wh @ h + nfft * noise_var * identity,
                wh,
            )

            x_zf = y @ wzf.T
            x_mmse = y @ wmmse.T

            bits_zf = demodulate(
                x_zf.reshape(-1),
                modulation,
            )

            bits_mmse = demodulate(
                x_mmse.reshape(-1),
                modulation,
            )

            errors_zf += np.count_nonzero(bits_zf != bits)

            errors_mmse += np.count_nonzero(bits_mmse != bits)

            total_bits += bits.size

            stop_reason = get_stop_reason(
                errors_zf,
                errors_mmse,
                total_bits,
                target_errors,
                max_bits,
            )

            if stop_reason is not None:
                break

        ber_zf = errors_zf / total_bits
        ber_mmse = errors_mmse / total_bits

        stop_reason = (
            get_stop_reason(
                errors_zf,
                errors_mmse,
                total_bits,
                target_errors,
                max_bits,
            )
            or "max_bits_reached"
        )

        results.append(
            (
                snr_db,
                ber_zf,
                ber_mmse,
                total_bits,
                errors_zf,
                errors_mmse,
                stop_reason,
            )
        )

        print(
            f"SNR={snr_db:>3} dB | "
            f"ZF={ber_zf:.6f} | "
            f"MMSE={ber_mmse:.6f} | "
            f"Bits={total_bits}"
        )

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    csv_path = output / "ber_comparison.csv"

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)

        writer.writerow(
            [
                "SNR_dB",
                "ZF_BER",
                "MMSE_BER",
                "Bits",
                "ZF_Errors",
                "MMSE_Errors",
                "Stop_Reason",
            ]
        )

        writer.writerows(results)

    snr = np.array([row[0] for row in results])
    zf = np.array([row[1] for row in results])
    mmse = np.array([row[2] for row in results])

    zf_plot = np.where(zf > 0, zf, np.nan)
    mmse_plot = np.where(mmse > 0, mmse, np.nan)

    plt.figure(figsize=(8, 5))

    plt.semilogy(snr, zf_plot, "o-", label="ZF")

    plt.semilogy(snr, mmse_plot, "s-", label="MMSE")

    plt.xlabel("Time-domain Receive SNR (dB)")
    plt.ylabel("Bit Error Rate")
    plt.title("2x2 MIMO-OFDM: ZF vs MMSE (Perfect CSI)")

    plt.grid(True, which="both", alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        output / "ber_comparison.png",
        dpi=200,
    )

    plt.close()

    print("BER results saved to results/")


if __name__ == "__main__":
    main()

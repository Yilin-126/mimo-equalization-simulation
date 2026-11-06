"""Compare perfect and LS-estimated CSI in multipath MIMO-OFDM."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np

from scripts.run_multipath_ber import equalize_grid
from src.mimo_sim.csi_estimation import estimate_mimo_channel_ls
from src.mimo_sim.multipath import (
    apply_multipath_channel,
    create_multipath_taps,
    multipath_frequency_response,
)
from src.mimo_sim.random_control import create_frame_samples
from src.mimo_sim.receiver import demodulate, ofdm_demodulate
from src.mimo_sim.transmitter import (
    map_to_resource_grid,
    modulate,
    ofdm_modulate,
)

ROOT = Path(__file__).resolve().parents[1]


def generate_pilot_noise(
    seed: int,
    frame_index: int,
    shape: tuple[int, ...],
) -> np.ndarray:
    """Generate reproducible independent pilot noise innovations.

    Args:
        seed: Independent nonnegative pilot-noise seed.
        frame_index: Nonnegative OFDM frame index.
        shape: Pilot noise dimensions (K, Nr, Nt).

    Returns:
        Unit-variance complex Gaussian array.

    Raises:
        ValueError: If seed, frame index, or dimensions are invalid.
    """
    if (
        type(seed) is not int
        or type(frame_index) is not int
        or seed < 0
        or frame_index < 0
        or len(shape) != 3
        or any(type(value) is not int or value <= 0 for value in shape)
    ):
        raise ValueError("Invalid pilot noise parameters")

    rng = np.random.default_rng(np.random.SeedSequence([seed, frame_index]))
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2)


def main() -> None:
    """Evaluate perfect versus LS CSI using common MIMO-OFDM frames.

    The data transmission, physical channel, and receiver noise
    are identical across all four algorithm/CSI combinations.
    Pilot noise is independently seeded and reused across SNR
    points at the same frame index.

    Args:
        None. Loads configs/simulation.json.

    Returns:
        None. Writes BER and channel MSE CSV files and PNG plots.

    Raises:
        ValueError: If antenna, channel, or sampling settings
            are incompatible with this controlled experiment.
    """
    with (ROOT / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    sim = config["simulation"]
    perf = config["performance"]
    multipath = config["multipath_comparison"]
    settings = config["csi_comparison"]
    random_control = config["random_control"]

    if (
        settings["enabled"] is not True
        or settings["channel_model"] != "three_tap_multipath"
        or settings["noise_policy"] != "reference_receive_power"
        or random_control["enabled"] is not True
        or random_control["reuse_across_snr"] is not True
    ):
        raise ValueError("Invalid controlled CSI comparison settings")

    nfft = int(system["fft_size"])
    cp = int(system["cp_length"])
    k = int(system["num_subcarriers"])
    num_symbols = int(system["num_ofdm_symbols"])

    nt = int(config["transmitter"]["num_tx_antennas"])
    nr = int(config["receiver"]["num_rx_antennas"])
    modulation = config["transmitter"]["modulation"]

    if (
        (nt, nr) != (2, 2)
        or not 0 < k <= nfft
        or not 0 < cp <= nfft
        or num_symbols <= 0
        or modulation not in ("QPSK", "16QAM")
    ):
        raise ValueError("Invalid MIMO-OFDM configuration")

    delays = multipath["delays_samples"]
    powers_db = multipath["powers_db"]

    if len(delays) != len(powers_db) or not delays or max(delays) > cp:
        raise ValueError("Invalid multipath delay configuration")

    pilot_seed = settings["pilot_seed"]
    pilot_amplitude = float(settings["pilot_amplitude"])

    if type(pilot_seed) is not int or pilot_seed < 0:
        raise ValueError("Invalid pilot seed")

    if not np.isfinite(pilot_amplitude) or pilot_amplitude <= 0:
        raise ValueError("Invalid pilot amplitude")

    bits_per_symbol = {"QPSK": 2, "16QAM": 4}[modulation]
    bits_per_frame = num_symbols * k * nt * bits_per_symbol

    target_errors = int(perf["target_errors"])
    max_frames = int(perf["max_bits_per_snr"]) // bits_per_frame
    min_frames = int(settings["min_frames_per_snr"])

    if target_errors <= 0 or not 1 <= min_frames <= max_frames:
        raise ValueError("Invalid BER statistical limits")

    snr_step = int(sim["snr_step_db"])
    if snr_step <= 0:
        raise ValueError("SNR step must be positive")

    snr_values = np.arange(
        sim["snr_start_db"],
        sim["snr_stop_db"] + 1,
        snr_step,
    )

    if snr_values.size == 0:
        raise ValueError("Empty SNR range")

    start = (nfft - k) // 2
    csi_modes = ("Perfect CSI", "LS CSI")
    methods = ("ZF", "MMSE")

    ber_records = []
    mse_records = []

    for snr_index, snr_db in enumerate(snr_values):
        errors = {(csi, method): 0 for csi in csi_modes for method in methods}

        total_bits = 0
        mse_sum = 0.0
        mse_frames = 0
        stop_reason = "max_bits_reached"

        for frame_index in range(max_frames):
            samples = create_frame_samples(
                config=config,
                snr_index=snr_index,
                frame_index=frame_index,
                num_bits=bits_per_frame,
                noise_shape=(num_symbols, nfft + cp, nr),
                channel_shape=(len(delays), nr, nt),
            )

            bits = samples["bits"]
            data_symbols = modulate(bits, modulation).reshape(num_symbols, k, nt)

            grid = map_to_resource_grid(data_symbols, nfft, k)
            tx = ofdm_modulate(grid, cp)

            taps = create_multipath_taps(
                samples["channel"],
                delays,
                powers_db,
            )

            clean = apply_multipath_channel(tx, taps)

            signal_power = float(np.mean(np.abs(clean) ** 2))
            noise_variance = signal_power / (10 ** (float(snr_db) / 10))

            common_noise = np.sqrt(noise_variance) * samples["noise"]

            received_time = clean + common_noise
            rx_fft = ofdm_demodulate(received_time, cp, nfft)
            rx_centered = np.fft.fftshift(rx_fft, axes=1)
            y = rx_centered[:, start : start + k, :]

            response = multipath_frequency_response(taps, nfft)
            h_centered = np.fft.fftshift(response, axes=0)
            h_true = h_centered[start : start + k]

            pilot_noise = generate_pilot_noise(
                seed=pilot_seed,
                frame_index=frame_index,
                shape=h_true.shape,
            )

            h_ls = estimate_mimo_channel_ls(
                true_channel=h_true,
                pilot_noise=pilot_noise,
                time_noise_variance=noise_variance,
                fft_size=nfft,
                pilot_amplitude=pilot_amplitude,
            )

            mse_sum += float(np.mean(np.abs(h_ls - h_true) ** 2))
            mse_frames += 1

            for csi_name, h_used in (
                ("Perfect CSI", h_true),
                ("LS CSI", h_ls),
            ):
                estimated = equalize_grid(
                    received=y,
                    channel=h_used,
                    noise_variance=noise_variance,
                    fft_size=nfft,
                )

                for method in methods:
                    detected_bits = demodulate(
                        estimated[method].reshape(-1),
                        modulation,
                    )

                    errors[(csi_name, method)] += int(
                        np.count_nonzero(detected_bits != bits)
                    )

            total_bits += bits.size

            reached = all(value >= target_errors for value in errors.values())

            if frame_index + 1 >= min_frames and reached:
                stop_reason = "target_reached"
                break

        for csi_name in csi_modes:
            for method in methods:
                count = errors[(csi_name, method)]
                ber_records.append(
                    (
                        float(snr_db),
                        csi_name,
                        method,
                        count / total_bits,
                        total_bits,
                        count,
                        stop_reason,
                    )
                )

        mse_records.append((float(snr_db), mse_sum / mse_frames, mse_frames))

        print(
            f"SNR={snr_db:>3} dB | Bits={total_bits:>8} | "
            f"Perfect ZF={errors[('Perfect CSI', 'ZF')] / total_bits:.6f} | "
            f"Perfect MMSE={errors[('Perfect CSI', 'MMSE')] / total_bits:.6f} | "
            f"LS ZF={errors[('LS CSI', 'ZF')] / total_bits:.6f} | "
            f"LS MMSE={errors[('LS CSI', 'MMSE')] / total_bits:.6f} | "
            f"MSE={mse_sum / mse_frames:.4e} | {stop_reason}"
        )

    output = ROOT / "results"
    output.mkdir(parents=True, exist_ok=True)

    with (output / "csi_ber_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "SNR_dB",
                "CSI",
                "Method",
                "BER",
                "Bits",
                "Errors",
                "Stop_Reason",
            ]
        )
        writer.writerows(ber_records)

    with (output / "csi_mse.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["SNR_dB", "LS_MSE", "Frames"])
        writer.writerows(mse_records)

    fig, ax = plt.subplots(figsize=(9, 5.5))

    for csi_name in csi_modes:
        for method in methods:
            selected = [
                row for row in ber_records if row[1] == csi_name and row[2] == method
            ]

            x = np.array([row[0] for row in selected])
            ber = np.array([row[3] for row in selected])
            visible = np.where(ber > 0, ber, np.nan)

            ax.semilogy(
                x,
                visible,
                marker="o" if method == "ZF" else "s",
                linestyle="-" if csi_name == "Perfect CSI" else "--",
                label=f"{csi_name} / {method}",
            )

    ax.set_xlabel("Time-domain receive SNR (dB)")
    ax.set_ylabel("Measured BER")
    ax.set_title("Multipath MIMO-OFDM: Perfect vs LS CSI")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "csi_ber_comparison.png", dpi=200)
    plt.close(fig)

    mse_x = np.array([row[0] for row in mse_records])
    mse_y = np.array([row[1] for row in mse_records])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.semilogy(mse_x, mse_y, "o-")
    ax.set_xlabel("Time-domain receive SNR (dB)")
    ax.set_ylabel("Channel estimation MSE")
    ax.set_title("LS MIMO Channel Estimation Error")
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(output / "csi_mse.png", dpi=200)
    plt.close(fig)

    print("\nSaved: results/csi_ber_comparison.csv")
    print("Saved: results/csi_ber_comparison.png")
    print("Saved: results/csi_mse.csv")
    print("Saved: results/csi_mse.png")


if __name__ == "__main__":
    main()

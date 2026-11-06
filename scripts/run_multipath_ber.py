"""Controlled BER comparison of flat and multipath MIMO channels."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


def equalize_grid(
    received: np.ndarray,
    channel: np.ndarray,
    noise_variance: float,
    fft_size: int,
) -> dict[str, np.ndarray]:
    """Apply per-subcarrier ZF and MMSE equalization.

    Args:
        received: Active received resource grid, shape (S,K,Nr).
        channel: Active frequency response, shape (K,Nr,Nt).
        noise_variance: Time-domain complex noise variance.
        fft_size: FFT length used by the OFDM modulator.

    Returns:
        Dictionary containing ZF and MMSE equalized grids,
        each with shape (S,K,Nt).

    Raises:
        ValueError: If antenna or grid dimensions are invalid.
    """
    if (
        received.ndim != 3
        or channel.ndim != 3
        or received.shape[1] != channel.shape[0]
        or received.shape[2] != channel.shape[1]
        or channel.shape[1] < channel.shape[2]
    ):
        raise ValueError("Invalid received grid or channel dimensions")

    hermitian = channel.conj().swapaxes(-1, -2)
    identity = np.eye(channel.shape[2], dtype=np.complex128)

    zf_weights = np.linalg.pinv(channel)

    mmse_weights = np.linalg.solve(
        hermitian @ channel + fft_size * noise_variance * identity,
        hermitian,
    )

    zf = np.einsum(
        "ktr,skr->skt",
        zf_weights,
        received,
    )

    mmse = np.einsum(
        "ktr,skr->skt",
        mmse_weights,
        received,
    )

    return {"ZF": zf, "MMSE": mmse}


def main() -> None:
    """Run a controlled flat-versus-multipath BER experiment.

    Args:
        None. Reads configs/simulation.json.

    Returns:
        None. Saves CSV data and a PNG figure to results/.

    Raises:
        ValueError: If the OFDM, channel, or comparison
            settings are invalid.
        KeyError: If required configuration entries are missing.
    """
    with (ROOT / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    sim = config["simulation"]
    perf = config["performance"]
    settings = config["multipath_comparison"]
    random_control = config["random_control"]

    models = settings["models"]
    expected_models = [
        "flat_rayleigh",
        "three_tap_multipath",
    ]

    if (
        settings["enabled"] is not True
        or models != expected_models
        or settings["reference_model"] != "flat_rayleigh"
        or settings["noise_policy"] != "reference_receive_power"
        or random_control["enabled"] is not True
        or random_control["reuse_realizations"] is not True
        or random_control["reuse_across_snr"] is not True
    ):
        raise ValueError("Invalid controlled multipath settings")

    k = int(system["num_subcarriers"])
    nfft = int(system["fft_size"])
    cp = int(system["cp_length"])
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
        raise ValueError("Invalid 2x2 MIMO-OFDM configuration")

    delays = settings["delays_samples"]
    powers_db = settings["powers_db"]

    if (
        len(delays) == 0
        or len(delays) != len(powers_db)
        or any(type(d) is not int or d < 0 for d in delays)
        or len(set(delays)) != len(delays)
        or max(delays) > cp
    ):
        raise ValueError("Multipath delays must fit within CP")

    bits_per_symbol = {
        "QPSK": 2,
        "16QAM": 4,
    }[modulation]

    bits_per_frame = num_symbols * k * nt * bits_per_symbol

    target_errors = int(perf["target_errors"])
    max_bits = int(perf["max_bits_per_snr"])
    max_frames = max_bits // bits_per_frame
    min_frames = int(settings["min_frames_per_snr"])

    if target_errors <= 0 or not 1 <= min_frames <= max_frames:
        raise ValueError("Invalid BER stopping configuration")

    snr_values = np.arange(
        sim["snr_start_db"],
        sim["snr_stop_db"] + 1,
        sim["snr_step_db"],
    )

    if sim["snr_step_db"] <= 0 or len(snr_values) == 0:
        raise ValueError("Invalid SNR range")

    start = (nfft - k) // 2
    records = []

    for snr_index, snr_db in enumerate(snr_values):
        errors = {model: {"ZF": 0, "MMSE": 0} for model in models}

        total_bits = 0
        stop_reason = "max_bits_reached"

        for frame_index in range(max_frames):
            samples = create_frame_samples(
                config=config,
                snr_index=snr_index,
                frame_index=frame_index,
                num_bits=bits_per_frame,
                noise_shape=(
                    num_symbols,
                    nfft + cp,
                    nr,
                ),
                channel_shape=(len(delays), nr, nt),
            )

            bits = samples["bits"]

            symbols = modulate(
                bits,
                modulation,
            ).reshape(num_symbols, k, nt)

            grid = map_to_resource_grid(
                symbols,
                nfft,
                k,
            )

            tx = ofdm_modulate(grid, cp)

            innovations = samples["channel"]

            # Both models originate from shared random innovations.
            flat_taps = innovations[:1].copy()

            multipath_taps = create_multipath_taps(
                innovations,
                delays,
                powers_db,
            )

            taps_by_model = {
                "flat_rayleigh": flat_taps,
                "three_tap_multipath": multipath_taps,
            }

            reference_clean = apply_multipath_channel(
                tx,
                flat_taps,
            )

            reference_power = float(np.mean(np.abs(reference_clean) ** 2))

            noise_variance = reference_power / (10 ** (float(snr_db) / 10))

            # Same noise realization and variance for both models.
            common_noise = np.sqrt(noise_variance) * samples["noise"]

            for model in models:
                taps = taps_by_model[model]

                clean = apply_multipath_channel(
                    tx,
                    taps,
                )

                rx = clean + common_noise

                rx_grid = ofdm_demodulate(
                    rx,
                    cp,
                    nfft,
                )

                centered = np.fft.fftshift(
                    rx_grid,
                    axes=1,
                )

                y = centered[:, start : start + k, :]

                frequency_channel = multipath_frequency_response(
                    taps,
                    nfft,
                )

                centered_channel = np.fft.fftshift(
                    frequency_channel,
                    axes=0,
                )

                h_active = centered_channel[start : start + k]

                estimated = equalize_grid(
                    y,
                    h_active,
                    noise_variance,
                    nfft,
                )

                for method, values in estimated.items():
                    recovered = demodulate(
                        values.reshape(-1),
                        modulation,
                    )

                    errors[model][method] += int(np.count_nonzero(recovered != bits))

            total_bits += bits.size

            reached = all(
                count >= target_errors
                for methods in errors.values()
                for count in methods.values()
            )

            if frame_index + 1 >= min_frames and reached:
                stop_reason = "target_reached"
                break

        for model in models:
            for method in ("ZF", "MMSE"):
                count = errors[model][method]

                records.append(
                    (
                        model,
                        float(snr_db),
                        method,
                        count / total_bits,
                        total_bits,
                        count,
                        stop_reason,
                    )
                )

        message = " | ".join(
            f"{model}: "
            f"ZF={errors[model]['ZF'] / total_bits:.6f}, "
            f"MMSE={errors[model]['MMSE'] / total_bits:.6f}"
            for model in models
        )

        print(
            f"SNR={snr_db:>3} dB | "
            f"Bits={total_bits:>8} | "
            f"{message} | {stop_reason}"
        )

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    csv_path = output / "multipath_ber_comparison.csv"

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "Model",
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

    for model in models:
        for method in ("ZF", "MMSE"):
            subset = [row for row in records if row[0] == model and row[2] == method]

            snr = np.array([row[1] for row in subset])
            ber = np.array([row[3] for row in subset])

            visible = np.where(
                ber > 0,
                ber,
                np.nan,
            )

            ax.semilogy(
                snr,
                visible,
                marker="o" if method == "ZF" else "s",
                linestyle="-" if method == "ZF" else "--",
                label=f"{model} / {method}",
            )

    ax.set_xlabel("Reference-channel receive SNR (dB)")
    ax.set_ylabel("Measured BER")
    ax.set_title("2x2 MIMO-OFDM: Flat vs Multipath Rayleigh")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()

    image_path = output / "multipath_ber_comparison.png"
    fig.savefig(image_path, dpi=200)
    plt.close(fig)

    print(f"\nSaved: {csv_path}")
    print(f"Saved: {image_path}")


if __name__ == "__main__":
    main()

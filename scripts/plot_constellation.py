"""Plot 2x2 MIMO QPSK constellation before and after equalization."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.mimo_sim.channel import add_awgn
from src.mimo_sim.receiver import ofdm_demodulate
from src.mimo_sim.transmitter import (
    generate_bits,
    map_to_resource_grid,
    modulate,
    ofdm_modulate,
)

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    with (ROOT / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    perf = config["performance"]

    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    h = np.array(perf["channel_real"]) + 1j * np.array(perf["channel_imag"])

    nt = config["transmitter"]["num_tx_antennas"]
    nr = config["receiver"]["num_rx_antennas"]
    s = system["num_ofdm_symbols"]

    if h.shape != (nr, nt) or (nr, nt) != (2, 2):
        raise ValueError("Expected a 2x2 MIMO channel")

    bits = generate_bits(2 * k * s * nt, rng)
    symbols = modulate(bits, "QPSK").reshape(s, k, nt)

    grid = map_to_resource_grid(symbols, nfft, k)
    tx = ofdm_modulate(grid, cp)
    clean = np.einsum("rt,slt->slr", h, tx)

    snr_db = config["channel"]["snr_db"]
    noise_var = np.mean(np.abs(clean) ** 2) / (10 ** (snr_db / 10))
    rx = add_awgn(clean, noise_var, rng)

    rx_grid = ofdm_demodulate(rx, cp, nfft)
    centered = np.fft.fftshift(rx_grid, axes=1)
    start = (nfft - k) // 2
    received = centered[:, start : start + k, :]

    wzf = np.linalg.pinv(h)
    wh = h.conj().T
    wmmse = np.linalg.solve(
        wh @ h + nfft * noise_var * np.eye(nt),
        wh,
    )

    zf = received @ wzf.T
    mmse = received @ wmmse.T

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    datasets = [
        (received[:, :, 0], "Before Equalization (Rx 1)"),
        (zf[:, :, 0], "After ZF (Stream 1)"),
        (mmse[:, :, 0], "After MMSE (Stream 1)"),
    ]

    for ax, (data, title) in zip(axes, datasets):
        points = data.reshape(-1)
        ax.scatter(points.real, points.imag, s=4, alpha=0.3)
        ax.set_title(title)
        ax.set_xlabel("In-phase")
        ax.set_ylabel("Quadrature")
        ax.grid(True, alpha=0.3)
        ax.set_aspect("equal")
        ax.set_xlim(-2, 2)
        ax.set_ylim(-2, 2)

    fig.suptitle(f"2x2 MIMO-OFDM QPSK Constellation ({snr_db} dB, Perfect CSI)")
    fig.tight_layout()

    output = ROOT / "results"
    output.mkdir(exist_ok=True)

    fig.savefig(output / "constellation_comparison.png", dpi=200)
    plt.close(fig)

    print("Constellation saved to results/constellation_comparison.png")


if __name__ == "__main__":
    main()

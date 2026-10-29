"""SISO-OFDM end-to-end integration test."""

import json
from pathlib import Path

import numpy as np
import pytest
from src.mimo_sim.receiver import demodulate, ofdm_demodulate
from src.mimo_sim.transmitter import (
    generate_bits,
    map_to_resource_grid,
    modulate,
    ofdm_modulate,
)
from src.mimo_sim.channel import add_awgn
from src.mimo_sim.channel_estimation import ls_channel_estimate
from src.mimo_sim.metrics import calculate_ber, calculate_mse
from src.mimo_sim.equalizer import mmse_equalize, zf_equalize


def test_siso_noiseless_link() -> None:
    """Verify noiseless SISO-OFDM transmission."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    bits = generate_bits(2 * k, rng)
    symbols = modulate(bits, "QPSK").reshape(1, k, 1)

    tx_grid = map_to_resource_grid(symbols, nfft, k)
    tx_signal = ofdm_modulate(tx_grid, cp)

    channel = np.array([[1 + 0j]])
    rx_signal = tx_signal * channel[0, 0]

    rx_grid = ofdm_demodulate(rx_signal, cp, nfft)
    centered = np.fft.fftshift(rx_grid, axes=1)
    start = (nfft - k) // 2
    received = centered[0, start : start + k, 0]

    estimated = np.array(
        [zf_equalize(np.array([value]), channel)[0] for value in received]
    )

    recovered_bits = demodulate(estimated, "QPSK")

    assert rx_grid.shape == tx_grid.shape
    assert estimated.shape == (k,)
    np.testing.assert_allclose(estimated, symbols.reshape(-1), atol=1e-10)
    assert calculate_ber(bits, recovered_bits) == 0.0


def test_siso_awgn_link() -> None:
    """Verify SISO-OFDM transmission over AWGN."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]
    snr_db = config["channel"]["snr_db"]

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    num_symbols = system["num_ofdm_symbols"]
    bits = generate_bits(2 * k * num_symbols, rng)
    symbols = modulate(bits, "QPSK").reshape(num_symbols, k, 1)

    tx_grid = map_to_resource_grid(symbols, nfft, k)
    tx_signal = ofdm_modulate(tx_grid, cp)

    channel = np.array([[0.8 + 0.3j]])
    channel_output = tx_signal * channel[0, 0]

    signal_power = np.mean(np.abs(channel_output) ** 2)
    noise_variance = signal_power / (10 ** (snr_db / 10))

    rx_signal = add_awgn(channel_output, noise_variance, rng)
    rx_grid = ofdm_demodulate(rx_signal, cp, nfft)

    centered = np.fft.fftshift(rx_grid, axes=1)
    start = (nfft - k) // 2
    received = centered[:, start : start + k, 0].reshape(-1)

    estimated = np.array(
        [zf_equalize(np.array([value]), channel)[0] for value in received]
    )

    recovered_bits = demodulate(estimated, "QPSK")
    ber = calculate_ber(bits, recovered_bits)

    assert recovered_bits.shape == bits.shape
    assert 0.0 <= ber <= 1.0
    assert ber < 0.1


def test_siso_pilot_channel_estimation() -> None:
    """Verify pilot-based SISO channel estimation."""
    rng = np.random.default_rng(42)

    num_pilots = 1200
    reference_pilots = np.ones(num_pilots, dtype=np.complex128)

    true_channel = 0.8 + 0.3j
    received_pilots = true_channel * reference_pilots

    estimated = ls_channel_estimate(received_pilots, reference_pilots)

    assert estimated.shape == (num_pilots,)

    np.testing.assert_allclose(
        estimated,
        true_channel,
        atol=1e-12,
    )


def test_siso_pilot_assisted_link() -> None:
    """Verify OFDM reception using pilot-based LS estimation."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    bits = generate_bits(2 * k, rng)
    data = modulate(bits, "QPSK").reshape(1, k, 1)
    pilots = np.ones((1, k, 1), dtype=np.complex128)

    symbols = np.concatenate([pilots, data], axis=0)
    tx_grid = map_to_resource_grid(symbols, nfft, k)
    tx_signal = ofdm_modulate(tx_grid, cp)

    true_channel = 0.8 + 0.3j
    rx_signal = tx_signal * true_channel

    rx_grid = ofdm_demodulate(rx_signal, cp, nfft)
    centered = np.fft.fftshift(rx_grid, axes=1)
    start = (nfft - k) // 2
    active = centered[:, start : start + k, 0]

    pilot_estimates = ls_channel_estimate(active[0], pilots[0, :, 0])
    estimated_channel = np.mean(pilot_estimates)

    channel_matrix = np.array([[estimated_channel]])
    equalized = np.array(
        [zf_equalize(np.array([value]), channel_matrix)[0] for value in active[1]]
    )

    recovered_bits = demodulate(equalized, "QPSK")

    np.testing.assert_allclose(estimated_channel, true_channel, atol=1e-12)
    assert calculate_ber(bits, recovered_bits) == 0.0


def test_siso_pilot_assisted_awgn() -> None:
    """Verify pilot-based LS estimation over AWGN."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]
    rng = np.random.default_rng(config["simulation"]["random_seed"])

    bits = generate_bits(2 * k, rng)
    data = modulate(bits, "QPSK").reshape(1, k, 1)
    pilots = np.ones_like(data)
    tx_symbols = np.concatenate([pilots, data], axis=0)

    tx_grid = map_to_resource_grid(tx_symbols, nfft, k)
    tx_signal = ofdm_modulate(tx_grid, cp)

    true_channel = 0.8 + 0.3j
    channel_output = tx_signal * true_channel

    snr_db = config["channel"]["snr_db"]
    signal_power = np.mean(np.abs(channel_output) ** 2)
    noise_variance = signal_power / (10 ** (snr_db / 10))

    rx_signal = add_awgn(channel_output, noise_variance, rng)
    rx_grid = ofdm_demodulate(rx_signal, cp, nfft)

    centered = np.fft.fftshift(rx_grid, axes=1)
    start = (nfft - k) // 2
    active = centered[:, start : start + k, 0]

    pilot_estimates = ls_channel_estimate(active[0], pilots[0, :, 0])
    estimated_channel = np.mean(pilot_estimates)

    channel_mse = calculate_mse(
        np.array([true_channel]),
        np.array([estimated_channel]),
    )

    channel_matrix = np.array([[estimated_channel]])
    equalized = np.array(
        [zf_equalize(np.array([value]), channel_matrix)[0] for value in active[1]]
    )

    recovered_bits = demodulate(equalized, "QPSK")
    ber = calculate_ber(bits, recovered_bits)

    assert np.isfinite(channel_mse)
    assert channel_mse < 0.01
    assert 0 <= ber < 0.1


@pytest.mark.parametrize("equalizer", ["ZF", "MMSE"])
def test_mimo_2x2_noiseless_link(equalizer: str) -> None:
    """Verify noiseless 2x2 MIMO-OFDM signal recovery."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.json"

    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]

    num_tx = config["transmitter"]["num_tx_antennas"]
    num_rx = config["receiver"]["num_rx_antennas"]

    assert (num_tx, num_rx) == (2, 2)

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    bits = generate_bits(2 * k * num_tx, rng)
    symbols = modulate(bits, "QPSK").reshape(1, k, num_tx)

    tx_grid = map_to_resource_grid(symbols, nfft, k)
    tx_signal = ofdm_modulate(tx_grid, cp)

    channel = np.array(
        [
            [1.0 + 0.2j, 0.3 - 0.1j],
            [0.2 + 0.1j, 0.9 - 0.2j],
        ],
        dtype=np.complex128,
    )

    rx_signal = np.einsum("rt,slt->slr", channel, tx_signal)

    assert tx_signal.shape == (1, nfft + cp, num_tx)
    assert rx_signal.shape == (1, nfft + cp, num_rx)

    rx_grid = ofdm_demodulate(rx_signal, cp, nfft)

    centered = np.fft.fftshift(rx_grid, axes=1)
    start = (nfft - k) // 2
    received = centered[0, start : start + k, :]

    if equalizer == "ZF":
        estimated = np.array([zf_equalize(y, channel) for y in received])
    else:
        estimated = np.array([mmse_equalize(y, channel, 0.0) for y in received])

    assert received.shape == (k, num_rx)
    assert estimated.shape == (k, num_tx)

    np.testing.assert_allclose(
        estimated,
        symbols[0],
        rtol=0,
        atol=1e-10,
    )

    recovered_bits = demodulate(
        estimated.reshape(-1),
        "QPSK",
    )

    assert calculate_ber(bits, recovered_bits) == 0.0


@pytest.mark.parametrize("equalizer", ["ZF", "MMSE"])
def test_mimo_2x2_awgn_link(equalizer: str) -> None:
    """Verify 2x2 MIMO-OFDM reception with AWGN."""
    config_path = Path(__file__).resolve().parents[1] / "configs" / "simulation.json"
    with config_path.open(encoding="utf-8") as file:
        config = json.load(file)

    system = config["system"]
    k = system["num_subcarriers"]
    nfft = system["fft_size"]
    cp = system["cp_length"]
    nt = config["transmitter"]["num_tx_antennas"]
    nr = config["receiver"]["num_rx_antennas"]

    rng = np.random.default_rng(config["simulation"]["random_seed"])

    bits = generate_bits(2 * k * nt, rng)
    symbols = modulate(bits, "QPSK").reshape(1, k, nt)

    tx_grid = map_to_resource_grid(symbols, nfft, k)
    tx_signal = ofdm_modulate(tx_grid, cp)

    channel = np.array(
        [
            [1.0 + 0.2j, 0.3 - 0.1j],
            [0.2 + 0.1j, 0.9 - 0.2j],
        ],
        dtype=np.complex128,
    )

    assert channel.shape == (nr, nt)

    clean_signal = np.einsum("rt,slt->slr", channel, tx_signal)

    snr_db = config["channel"]["snr_db"]
    signal_power = np.mean(np.abs(clean_signal) ** 2)
    noise_variance = signal_power / (10 ** (snr_db / 10))

    received_signal = add_awgn(clean_signal, noise_variance, rng)
    received_grid = ofdm_demodulate(received_signal, cp, nfft)

    centered = np.fft.fftshift(received_grid, axes=1)
    start = (nfft - k) // 2
    received = centered[0, start : start + k, :]

    if equalizer == "ZF":
        estimated = np.array([zf_equalize(y, channel) for y in received])
    else:
        frequency_noise_variance = nfft * noise_variance
        estimated = np.array(
            [mmse_equalize(y, channel, frequency_noise_variance) for y in received]
        )

    recovered_bits = demodulate(estimated.reshape(-1), "QPSK")
    ber = calculate_ber(bits, recovered_bits)

    assert received.shape == (k, nr)
    assert estimated.shape == (k, nt)
    assert np.all(np.isfinite(estimated))
    assert 0.0 <= ber <= 1.0

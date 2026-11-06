"""Integration tests for controlled multi-scenario BER evaluation."""

import csv
import json
from pathlib import Path

import pytest

from scripts import run_scenario_ber


@pytest.fixture
def experiment_root(tmp_path: Path, monkeypatch) -> Path:
    """Create an isolated, lightweight experiment environment.

    Args:
        tmp_path: Temporary directory supplied by pytest.
        monkeypatch: Pytest fixture for replacing the project root.

    Returns:
        Temporary project root containing a test configuration.
    """
    original_root = run_scenario_ber.ROOT

    with (original_root / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    config["simulation"]["snr_start_db"] = 0
    config["simulation"]["snr_stop_db"] = 2
    config["simulation"]["snr_step_db"] = 2
    config["performance"]["max_bits_per_snr"] = 134400
    config["performance"]["target_errors"] = 200
    config["scenario_comparison"]["min_frames_per_snr"] = 2

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with (config_dir / "simulation.json").open("w", encoding="utf-8") as file:
        json.dump(config, file)

    monkeypatch.setattr(run_scenario_ber, "ROOT", tmp_path)

    return tmp_path


def test_scenario_ber_is_reproducible(experiment_root: Path) -> None:
    """Check reproducibility and fairness of BER comparisons.

    Args:
        experiment_root: Isolated experiment project directory.

    Returns:
        None. Assertions validate generated CSV results.
    """
    run_scenario_ber.main()

    csv_path = experiment_root / "results" / "scenario_ber_comparison.csv"
    first_output = csv_path.read_bytes()

    run_scenario_ber.main()
    second_output = csv_path.read_bytes()

    assert first_output == second_output

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 8

    expected_pairs = {
        ("independent", "ZF"),
        ("independent", "MMSE"),
        ("rx_correlated_0p7", "ZF"),
        ("rx_correlated_0p7", "MMSE"),
    }

    for snr in (0.0, 2.0):
        group = [row for row in rows if float(row["SNR_dB"]) == snr]

        assert len(group) == 4

        actual_pairs = {(row["Scenario"], row["Method"]) for row in group}
        assert actual_pairs == expected_pairs

        bit_counts = {int(row["Bits"]) for row in group}
        assert len(bit_counts) == 1

        for row in group:
            bits = int(row["Bits"])
            errors = int(row["Errors"])
            ber = float(row["BER"])

            assert bits > 0
            assert 0 <= errors <= bits
            assert ber == pytest.approx(errors / bits)

            assert row["Stop_Reason"] in {
                "target_reached",
                "max_bits_reached",
            }

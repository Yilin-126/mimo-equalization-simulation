"""Integration tests for controlled multipath MIMO-OFDM BER."""

import csv
import json
from pathlib import Path

import pytest

from scripts import run_multipath_ber


@pytest.fixture
def experiment_root(tmp_path: Path, monkeypatch) -> Path:
    """Create an isolated multipath BER experiment.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Fixture used to redirect output files.

    Returns:
        Temporary project directory with a test configuration.
    """
    with (run_multipath_ber.ROOT / "configs" / "simulation.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    config["simulation"]["snr_start_db"] = 0
    config["simulation"]["snr_stop_db"] = 2
    config["simulation"]["snr_step_db"] = 2

    config["performance"]["target_errors"] = 200
    config["performance"]["max_bits_per_snr"] = 134400
    config["multipath_comparison"]["min_frames_per_snr"] = 2

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with (config_dir / "simulation.json").open("w", encoding="utf-8") as file:
        json.dump(config, file)

    monkeypatch.setattr(run_multipath_ber, "ROOT", tmp_path)

    return tmp_path


def test_multipath_ber_reproducibility_and_fairness(
    experiment_root: Path,
) -> None:
    """Verify repeatability and shared BER sampling budgets.

    Args:
        experiment_root: Isolated experiment directory.

    Returns:
        None. Assertions check CSV results and generated plots.
    """
    run_multipath_ber.main()

    results = experiment_root / "results"
    csv_path = results / "multipath_ber_comparison.csv"
    image_path = results / "multipath_ber_comparison.png"

    assert csv_path.exists()
    assert image_path.exists()

    first_output = csv_path.read_bytes()

    run_multipath_ber.main()

    assert csv_path.read_bytes() == first_output

    with csv_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 8

    expected_pairs = {
        ("flat_rayleigh", "ZF"),
        ("flat_rayleigh", "MMSE"),
        ("three_tap_multipath", "ZF"),
        ("three_tap_multipath", "MMSE"),
    }

    for snr in (0.0, 2.0):
        group = [row for row in rows if float(row["SNR_dB"]) == snr]

        assert len(group) == 4

        assert {(row["Model"], row["Method"]) for row in group} == expected_pairs

        bit_counts = {int(row["Bits"]) for row in group}
        assert bit_counts == {134400}

        for row in group:
            bits = int(row["Bits"])
            errors = int(row["Errors"])
            ber = float(row["BER"])

            assert 0 <= errors <= bits
            assert ber == pytest.approx(errors / bits)

            assert row["Stop_Reason"] in {
                "target_reached",
                "max_bits_reached",
            }

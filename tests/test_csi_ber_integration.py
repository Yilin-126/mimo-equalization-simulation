"""Integration tests for Perfect CSI and LS CSI BER evaluation."""

import csv
import json
from pathlib import Path

import pytest

from scripts import run_csi_ber


@pytest.fixture
def experiment_root(tmp_path: Path, monkeypatch) -> Path:
    """Create a lightweight isolated CSI experiment.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Fixture used to redirect experiment outputs.

    Returns:
        Temporary project root containing simulation configuration.
    """
    with (run_csi_ber.ROOT / "configs" / "simulation.json").open(
        encoding="utf-8"
    ) as file:
        config = json.load(file)

    config["simulation"]["snr_start_db"] = 0
    config["simulation"]["snr_stop_db"] = 2
    config["simulation"]["snr_step_db"] = 2

    config["performance"]["target_errors"] = 200
    config["performance"]["max_bits_per_snr"] = 134400
    config["csi_comparison"]["min_frames_per_snr"] = 2

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with (config_dir / "simulation.json").open("w", encoding="utf-8") as file:
        json.dump(config, file)

    monkeypatch.setattr(run_csi_ber, "ROOT", tmp_path)

    return tmp_path


def test_csi_ber_reproducibility_and_fairness(
    experiment_root: Path,
) -> None:
    """Verify CSI experiment outputs and common sample budgets.

    Args:
        experiment_root: Isolated configuration and output directory.

    Returns:
        None. Assertions check repeatability, BER accounting,
        and channel estimation MSE.
    """
    run_csi_ber.main()

    result_dir = experiment_root / "results"
    ber_path = result_dir / "csi_ber_comparison.csv"
    mse_path = result_dir / "csi_mse.csv"

    assert (result_dir / "csi_ber_comparison.png").exists()
    assert (result_dir / "csi_mse.png").exists()

    first_ber = ber_path.read_bytes()
    first_mse = mse_path.read_bytes()

    run_csi_ber.main()

    assert ber_path.read_bytes() == first_ber
    assert mse_path.read_bytes() == first_mse

    with ber_path.open(newline="", encoding="utf-8") as file:
        ber_rows = list(csv.DictReader(file))

    with mse_path.open(newline="", encoding="utf-8") as file:
        mse_rows = list(csv.DictReader(file))

    assert len(ber_rows) == 8
    assert len(mse_rows) == 2

    expected_pairs = {
        ("Perfect CSI", "ZF"),
        ("Perfect CSI", "MMSE"),
        ("LS CSI", "ZF"),
        ("LS CSI", "MMSE"),
    }

    for snr in (0.0, 2.0):
        group = [row for row in ber_rows if float(row["SNR_dB"]) == snr]

        assert len(group) == 4

        assert {(row["CSI"], row["Method"]) for row in group} == expected_pairs

        assert len({int(row["Bits"]) for row in group}) == 1

        for row in group:
            bits = int(row["Bits"])
            errors = int(row["Errors"])
            ber = float(row["BER"])

            assert bits == 134400
            assert 0 <= errors <= bits
            assert ber == pytest.approx(errors / bits)

            assert row["Stop_Reason"] in {
                "target_reached",
                "max_bits_reached",
            }

    mse_by_snr = {float(row["SNR_dB"]): float(row["LS_MSE"]) for row in mse_rows}

    assert mse_by_snr[0.0] > mse_by_snr[2.0] > 0

    for row in mse_rows:
        assert int(row["Frames"]) == 2

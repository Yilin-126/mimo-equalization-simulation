"""Integration test for common random samples across SNR points."""

import json
from pathlib import Path

import numpy as np

from scripts import run_scenario_ber
from src.mimo_sim.random_control import create_frame_samples


def test_snr_reuse_in_full_simulation(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Verify that the scenario BER runner reuses base samples.

    Args:
        tmp_path: Temporary directory provided by pytest.
        monkeypatch: Pytest fixture for isolating configuration
            and capturing frame-sample generation.

    Returns:
        None. Assertions verify cross-SNR sample equality.
    """
    original_root = run_scenario_ber.ROOT

    with (original_root / "configs" / "simulation.json").open(encoding="utf-8") as file:
        config = json.load(file)

    # Use two SNR points and a small budget for a fast test.
    config["simulation"]["snr_start_db"] = 0
    config["simulation"]["snr_stop_db"] = 2
    config["simulation"]["snr_step_db"] = 2

    config["performance"]["max_bits_per_snr"] = 134400
    config["performance"]["target_errors"] = 200
    config["scenario_comparison"]["min_frames_per_snr"] = 2

    config["random_control"]["reuse_realizations"] = True
    config["random_control"]["reuse_across_snr"] = True

    config_dir = tmp_path / "configs"
    config_dir.mkdir()

    with (config_dir / "simulation.json").open("w", encoding="utf-8") as file:
        json.dump(config, file)

    captured = {}

    def capture_samples(**kwargs) -> dict:
        """Capture samples used by the actual BER simulation.

        Args:
            **kwargs: Arguments passed to create_frame_samples.

        Returns:
            Generated bit, noise, and channel samples.
        """
        samples = create_frame_samples(**kwargs)

        key = (
            kwargs["snr_index"],
            kwargs["frame_index"],
        )

        captured[key] = {name: values.copy() for name, values in samples.items()}

        return samples

    monkeypatch.setattr(run_scenario_ber, "ROOT", tmp_path)
    monkeypatch.setattr(
        run_scenario_ber,
        "create_frame_samples",
        capture_samples,
    )

    run_scenario_ber.main()

    for frame_index in (0, 1):
        first = captured[(0, frame_index)]
        second = captured[(1, frame_index)]

        for name in ("bits", "noise", "channel"):
            assert np.array_equal(
                first[name],
                second[name],
            )

    assert len(captured) == 4

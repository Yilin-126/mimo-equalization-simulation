"""MIMO-OFDM链路仿真主程序。"""

import json
from pathlib import Path


def load_config() -> dict:
    """读取独立仿真配置文件。

    Returns:
        包含各模块仿真参数的配置字典。
    """
    project_root = Path(__file__).resolve().parent
    config_path = project_root / "configs" / "simulation.json"

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    """加载仿真参数并验证配置。"""
    config = load_config()

    print("=== MIMO-OFDM Simulation ===")
    print(f"Subcarriers: {config['system']['num_subcarriers']}")
    print(f"OFDM symbols: {config['system']['num_ofdm_symbols']}")
    print(f"Modulation: {config['transmitter']['modulation']}")
    print(
        f"Antenna configuration: "
        f"{config['transmitter']['num_tx_antennas']}x"
        f"{config['receiver']['num_rx_antennas']}"
    )
    print(f"Channel: {config['channel']['model']}")
    print(f"Equalizer: {config['receiver']['equalizer']}")
    print(f"SNR: {config['channel']['snr_db']} dB")


if __name__ == "__main__":
    main()

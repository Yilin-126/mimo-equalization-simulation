# 实验结果索引清单 (Experiment Results Index)

本目录保存了所有仿真的原始数据与高清图表。所有数据均基于 `configs/simulation.json` 生成。

| 对应图表 | 实验内容 | 原始数据路径 | 复现命令 | 备注 |
| :--- | :--- | :--- | :--- | :--- |
| **报告图 7 / 图 8** | Perfect vs LS CSI，以及 LS MSE | `results/csi_ber_comparison.csv`<br>`results/csi_mse.csv` | `python -m scripts.run_csi_ber` | 三抽头多径，含导频噪声模型 |
| **报告图 6** | 平坦 vs 三抽头多径 BER | `results/multipath_ber_comparison.csv` | `python -m scripts.run_multipath_ber` | 共同样本对比 |
| **报告图 3** | C vs Python 加速比与数值一致性 | `results/c_python_benchmark.csv`<br>`results/c_python_error_comparison.csv` | `python -m scripts.benchmark_c_python`<br>`python -m scripts.check_c_equalizer` | C模块需先编译 `equalizer.dll` |
| **新增表 / 图 10** | 空间相关对条件数的影响 | `results/condition_number_stats.csv` | `python -m scripts.analyze_condition_number` | 统计 20000 次信道实现，固定种子 |
| **新增表** | LS 相对 Perfect 的 BER 恶化倍数 | `results/ber_gap_analysis.csv` | `python -m scripts.analyze_ber_gap` | 支撑 MMSE 正则化鲁棒性分析 |
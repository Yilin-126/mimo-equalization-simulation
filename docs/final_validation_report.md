# MIMO-OFDM项目最终验证报告

## 1. 完成范围

完成Python MIMO-OFDM收发链路、ZF/MMSE、C语言1×1/2×2均衡、独立/相关Rayleigh信道、三抽头多径模型以及Perfect/LS CSI对比。配置与实验均使用相对路径，支持可重复的随机样本控制。

## 2. 验证摘要

| 验证项 | 结果 |
|---|---|
| Python自动化测试 | **105 passed** |
| C/Python冻结向量 | 16次通过，最大误差`1.379e-10` |
| C/ctypes相对Python加速 | ZF约6.68–8.47倍，MMSE约2.95–3.57倍 |
| 原生C计算路径 | 1×1 ZF约16ns、2×2 ZF约52ns/次 |
| 独立/相关Rayleigh BER | 已运行并保存CSV、PNG |
| 平坦/三抽头多径BER | 已运行并保存CSV、PNG |
| Perfect/LS CSI及MSE | 已运行并保存CSV、PNG |

## 3. 代表性实测

**CSI实验：** 2×2 QPSK、三抽头多径、各SNR至少20帧，表中为本轮结果。

| SNR | Perfect ZF | Perfect MMSE | LS ZF | LS MMSE | LS MSE |
|---|---:|---:|---:|---:|---:|
| 0 dB | 0.2306 | 0.1701 | 0.3483 | 0.3040 | 1.0557 |
| 10 dB | 0.0531 | 0.0348 | 0.1212 | 0.0919 | 0.1056 |
| 20 dB | 0.0076 | 0.0045 | 0.0190 | 0.0136 | 0.01056 |

此实验中LS CSI的BER高于Perfect CSI，MMSE低于ZF；MSE随SNR上升而下降。多径实验在当前设置下表现为较高的BER，但两种信道使用参考信道噪声方差，场景间接收功率并非逐实现完全相等。

## 4. 复现与限制

```powershell
python -m pytest tests/ -q
python -m scripts.run_scenario_ber
python -m scripts.run_multipath_ber
python -m scripts.run_csi_ber
```

详见 `results/` 中的CSV/PNG、`docs/adaptive_ber_method.md`、`docs/c_python_validation.md`。项目使用自定义三抽头多径而非标准EPA/EVA；LS采用理想正交训练且未计NR DMRS开销；结论限于当前仿真和有限测试预算。

仓库：https://github.com/Yilin-126/mimo-equalization-simulation

# MIMO-OFDM链路仿真设计说明

## 1. 目标与结构

实现可重复的MIMO-OFDM链路，比较ZF/MMSE在信道相关性、多径及信道估计误差下的性能，并使用C版本与Python结果对齐。

链路：随机比特 → QPSK/16QAM → 资源网格 → IFFT及循环前缀 → MIMO信道及噪声 → 去循环前缀及FFT → ZF/MMSE → 硬判决 → BER。

## 2. 已实现模块

| 模块 | 功能 | 主要文件 |
|---|---|---|
| 收发 | 调制、OFDM、判决 | `src/mimo_sim/transmitter.py`、`receiver.py` |
| 基础信道 | AWGN、平坦Rayleigh | `channel.py` |
| 场景信道 | 独立/接收端相关Rayleigh（ρ=0.7） | `channel_scenarios.py` |
| 多径信道 | 三抽头时域卷积与频域响应 | `multipath.py` |
| 信道估计 | 基础LS、理想正交训练LS CSI | `channel_estimation.py`、`csi_estimation.py` |
| 均衡与指标 | Python ZF/MMSE、BER/EVM/MSE | `equalizer.py`、`metrics.py` |
| C均衡 | 1×1/2×2 ZF/MMSE | `c_src/equalizer.c` |
| 实验控制 | 随机种子、共同样本、统计停止 | `random_control.py`、`ber_stopping.py` |

## 3. 基准实验约定

主要参数：2×2、QPSK、FFT=2048、有效子载波=1200、CP=144、14个OFDM符号。每帧67,200个信息比特；多场景实验每SNR至少20帧、最多30帧，目标误码数200。

- `reuse_realizations=true`：不同场景共享基础发送比特、噪声和信道创新样本。
- `reuse_across_snr=true`：相同帧编号在不同SNR复用基础随机样本；SNR通过噪声幅度调整。
- 信道比较：同一帧采用参考信道接收功率计算噪声方差，并共用实际噪声；横轴是**参考信道目标接收SNR**，不代表各场景瞬时SNR相同。
- Perfect/LS CSI比较：真实信道、发送数据及数据噪声保持相同，仅均衡器所用信道矩阵不同。

自定义多径：延迟 `[0,2,5]` 个采样，功率 `[0,-3,-6] dB`；最大延迟不超过CP。多径时域卷积在各OFDM块内处理，适用于本次CP覆盖延迟的测试模型。

## 4. 实现范围与验证

实验：固定信道BER、独立/相关Rayleigh对比、平坦/三抽头多径对比、Perfect/LS CSI及MSE。输出为 `results/` 中的CSV与PNG。

最近一次验证：105项Python测试通过；C/Python 16项数值比较通过。统计数据和边界见 `docs/final_validation_report.md`。

**局限：** 相关信道仅实现简化2×2接收侧模型；多径不等于EPA/EVA；LS未包含标准NR DMRS映射、插值和导频开销；未覆盖硬件链路或超过CP的跨符号干扰。

# MIMO-OFDM 链路仿真方案设计说明

> 文档性质：工程设计说明｜状态：当前基线｜适用范围：本项目已实现功能

## 1. 编写目的与项目范围

本项目以简化的 MIMO-OFDM 物理层链路为对象，构建 Python 仿真平台，实现调制解调、OFDM 收发、信道模型、信道估计、ZF/MMSE 均衡和性能评价；同时开发小规模 C 均衡模块，检验跨语言数值一致性和执行耗时。

项目参考 5G NR 物理层相关概念，**并非完整的 3GPP NR 标准链路实现**。本文严格区分“已实现”“独立实现但尚未接入主流程”和“计划扩展”，所有结果以当前源码、测试及结果文件为准。

## 2. 实现范围与边界

| 类别 | 功能 | 当前状态 |
|---|---|---|
| 基础收发链路 | QPSK、16QAM、OFDM IFFT/FFT、CP、硬判决 | 已实现 |
| 信道与噪声 | AWGN、独立 Rayleigh MIMO、频率平坦信道网格 | 已实现 |
| 信道估计 | 导频 LS 估计、频域线性插值 | 独立模块已实现；主 BER 扫描未接入 |
| 均衡 | Python ZF/MMSE | 已实现 |
| C 均衡 | 1×1、2×2 方阵 ZF/MMSE | 已实现；未支持 4×4 或批量网格 API |
| 性能评价 | BER、归一化 RMS EVM、信道 MSE | 已实现 |
| 主 BER 实验 | 固定 2×2 信道、Perfect CSI、自适应停止 | 已实现 |
| 实验复现 | 比特与噪声独立随机流，固定信道重复运行 | 已验证；跨信道统一变量控制未完成 |
| 性能计时 | Python/C-ctypes 耗时、原生 C 循环耗时 | 已完成两类不同计时口径 |
| 扩展信道 | EPA/EVA、多径、空间相关、非理想 CSI 集成 | 尚未实现 |

## 3. 系统架构与主要数据流

**发送端：** 随机比特 → QPSK/16QAM 调制 → 有效子载波映射 → IFFT → 添加循环前缀（CP）。

**信道端：** 固定或独立生成的 Rayleigh MIMO 信道矩阵、AWGN；现有主 BER 扫描固定采用 `configs/simulation.json` 中的 2×2 复数信道矩阵。

**接收端：** 去 CP → FFT → 有效子载波提取 → ZF/MMSE 均衡 → 硬判决 → 误码累计与结果输出。

**重要边界：** 主 BER 扫描使用已知的真实信道矩阵（Perfect CSI）。LS 导频估计及插值虽然有独立实现，但尚未接入主 BER 扫描。`generate_frequency_channel()` 只是将同一频率平坦矩阵复制到资源网格，不属于多径信道模型。

## 4. 当前仿真基线参数

| 参数 | 基线值 |
|---|---|
| 有效子载波数 `K` | 1200 |
| FFT 点数 `Nfft` | 2048 |
| OFDM 符号数 `S` | 14 |
| CP 长度 `Ncp` | 144 个采样点 |
| 主 BER 调制方式 | QPSK |
| 发射/接收天线 | `Nt=2`，`Nr=2` |
| 信道与 CSI | 固定 2×2 复数矩阵、Perfect CSI |
| SNR 扫描 | 0～20 dB，间隔 2 dB |
| 目标误码数 | ZF 和 MMSE 各 200 个 |
| 每个 SNR 最大比特数 | 2,016,000 |
| 单帧统计比特数 | `14 × 1200 × 2 × 2 = 67,200` |

所有主要参数统一存放于 `configs/simulation.json`。主 BER 脚本按**时域接收信号平均功率**计算噪声方差：`noise_variance = signal_power / 10^(SNR_dB/10)`，MMSE 的频域正则项包含 `Nfft × noise_variance`。冻结 C/Python 参考向量使用的噪声方差定义应单独说明，不可混为同一 SNR 口径。

## 5. 软件模块与接口分工

| 文件 | 主要职责 |
|---|---|
| `src/mimo_sim/transmitter.py` | 比特生成、数字调制、资源网格映射、OFDM 发送 |
| `src/mimo_sim/channel.py` | AWGN、Rayleigh MIMO、频率平坦信道网格 |
| `src/mimo_sim/channel_estimation.py` | 导频 LS 估计、线性插值 |
| `src/mimo_sim/receiver.py` | OFDM 接收和硬判决 |
| `src/mimo_sim/equalizer.py` | Python ZF/MMSE 单向量均衡 |
| `src/mimo_sim/metrics.py` | BER、RMS EVM、MSE |
| `src/mimo_sim/ber_stopping.py` | 双算法目标误码与最大样本量停止条件 |
| `src/mimo_sim/random_control.py` | 比特、噪声、信道随机流初始化 |
| `scripts/run_ber_scan.py` | 固定信道 BER-SNR 扫描，输出 CSV/PNG |
| `scripts/check_c_equalizer.py` | C 与冻结 Python 输出对齐 |
| `scripts/benchmark_c_python.py` | Python 与 C/ctypes 逐向量计时 |
| `c_src/equalizer.c`、`c_src/equalizer.h` | C 1×1/2×2 线性均衡接口 |
| `c_src/benchmark_core.c` | Windows 原生 C 循环耗时测试 |

更细致的数据维度、内存和异常约定见 [`interfaces.md`](interfaces.md)。

## 6. 自适应 BER 统计及变量控制

ZF 和 MMSE 在每个 SNR 点共享**相同的发送比特、固定信道及同一接收含噪信号**。当两种算法均累积至少 200 个错误比特，或完整帧数达到最大样本预算后，停止该 SNR 点的仿真；输出 `Bits`、`ZF_Errors`、`MMSE_Errors`、`Stop_Reason`。

`random_control` 配置采用独立种子：`bit_seed=20261106`、`noise_seed=20261107`、`channel_seed=20261108`。主 BER 程序目前使用比特与噪声随机流；由于信道固定，暂不使用信道随机流。已有独立随机流的单元测试及两次扫描结果一致性验证。`enabled`、`reuse_realizations` 配置项尚未实现开关行为，因此**不能宣称不同信道模型之间的严格变量控制已经完成**。

高 SNR 下零观测误码仅代表有限测试样本中未发现错误。在独立误码近似假设下，零误码时单侧 95% 上界可估算为 `3/N`；当 `N=2,016,000` 时约为 `1.49×10^-6`。

具体算法、停止原因及统计限制见 [`adaptive_ber_method.md`](adaptive_ber_method.md)。

## 7. C 均衡实现与验证

C 接口使用 `double complex`，信道矩阵按行优先方式存储：`h[rx*n + tx]`；目前只允许 `n=1` 或 `n=2` 的方阵。ZF 求解 `H x_hat = y`，MMSE 求解 `(H^H H + sigma^2 I) x_hat = H^H y`，采用带部分主元选择的复数高斯消元。状态码：`0` 成功、`-1` 参数无效、`-2` 不可求解。

冻结参考数据包含 8 种输入场景，每种对比 ZF/MMSE，共 **16 项数值对齐全部通过**；最大绝对复数误差 `1.379×10^-10`，验收阈值 `1×10^-8`。这些数据只能证明已测试场景满足阈值，不能作为所有病态信道的误差上界。

## 8. 性能测试与计时边界

两类耗时测试均覆盖 1×1、2×2 两种天线配置，64、256、1200 个**独立均衡向量**以及 ZF/MMSE，共 12 组配置。

**Python 与 C/ctypes 调用路径比较：** 实测 ZF 加速比约 **6.68～8.47 倍**，MMSE 约 **2.95～3.57 倍**。该结果包含 ctypes、输出分配等调用开销，且两套内部求解步骤存在差异，不能视为编程语言本身的通用加速结论。

**原生 C 循环计时：** 每次调用耗时约为 1×1 ZF **16.115～16.313 ns**、1×1 MMSE **26.376～27.834 ns**、2×2 ZF **51.948～52.922 ns**、2×2 MMSE **85.808～87.397 ns**。这一口径不经过 Python，但包含 C 循环及函数调用开销。两类实验的输入生成方式与计时流程不同，不能直接将时间差解释为 ctypes 开销。

具体数据和适用边界见 [`c_equalizer_design.md`](c_equalizer_design.md) 与 [`c_python_validation.md`](c_python_validation.md)。

## 9. 验收结果、风险与后续事项

当前已验证：Python 自动化测试 **70 项通过**；冻结 C/Python 对齐 **16 项通过**；固定信道 BER 重复运行生成一致的 CSV；原生 C 与接口级耗时数据已保存。

主要风险与待办：相关/多径信道尚未实现，非理想 CSI 未集成到 BER 主流程；跨信道场景尚未复用同一组基础随机样本；C 尚未提供 4×4 和批处理 API；两类计时尚未统一到完全相同的输入和计时边界。

## 10. 文档维护要求

每次变更算法、配置、接口或测试，应在同一次 Git 提交中同步更新相关 `docs/*.md` 和 `README.md`。所有结论以**实际代码、测试输出与结果文件**为依据；尚未实现的内容必须明确标记为“计划”或“未实现”。



## 8. 多场景MIMO信道设计

### 8.1 已实现的信道模型

当前系统在原有AWGN及独立Rayleigh信道基础上，新增接收端空间相关Rayleigh平坦衰落模型。

| 信道模型 | 实现状态 | 说明 |
|---|---|---|
| AWGN | 已实现 | 复高斯白噪声 |
| 独立Rayleigh | 已实现 | 独立复高斯MIMO信道 |
| 接收端相关Rayleigh | 已实现 | 2×2接收端空间相关，ρ=0.7 |
| EPA/EVA多径 | 未实现 | 后续扩展 |
| 非理想CSI | 未集成 | 后续扩展 |

主要模块：

- `src/mimo_sim/channel.py`：基础信道及噪声模型
- `src/mimo_sim/channel_scenarios.py`：独立及相关Rayleigh场景
- `scripts/run_scenario_ber.py`：多场景BER性能评估

### 8.2 接收端相关信道模型

使用共同的独立Rayleigh信道矩阵H0，通过相关性变换得到：

`H_corr = L × H0`

其中：

`L = [[1, 0], [ρ, sqrt(1-ρ²)]]`

当前采用ρ=0.7。

两种场景复用相同的基础信道随机样本，确保差异来自信道相关性变换。

## 9. 实验变量控制设计

随机数管理模块：

`src/mimo_sim/random_control.py`

使用三个独立随机种子，分别控制：

- 发送比特
- 接收噪声
- 基础信道

### 9.1 场景间复用

当 `reuse_realizations=true` 时，不同场景共享相同基础随机样本。

当其为 `false` 时，不同场景使用独立但可重复的随机流。

### 9.2 跨SNR复用

当 `reuse_across_snr=true` 时，同一帧编号在不同SNR点复用相同的发送比特、基础噪声和基础信道。

SNR变化通过调整噪声幅度实现。

### 9.3 多场景共同停止

两种信道及ZF/MMSE算法使用相同的统计帧数。

当前配置最少20帧，最多30帧，每帧67,200比特。

使用参考信道接收功率确定噪声方差，避免不同信道场景独立调整噪声幅度。

## 10. 多场景性能验证

当前实验包括：

- 2×2 MIMO-OFDM
- QPSK
- 独立Rayleigh与接收端相关Rayleigh
- ZF与MMSE
- Perfect CSI
- SNR范围0–20 dB

本轮实验中，各SNR点统计20帧，共1,344,000比特。

在当前测试条件下：

- MMSE的BER低于ZF。
- 接收端相关信道的BER高于独立信道。
- BER随SNR提高整体呈下降趋势。

结果文件：

- `results/scenario_ber_comparison.csv`
- `results/scenario_ber_comparison.png`

目前实验仍采用简化的频率平坦衰落信道，尚未实现EPA/EVA多径及非理想CSI性能比较。

## 11. 工程验证状态

当前项目已完成：

- Python自动化测试：93项通过
- C/Python冻结参考向量比较：16项通过
- C/Python接口级性能测试
- 原生C计算路径性能测试
- 多场景共享随机样本验证
- 跨SNR随机样本复用集成验证

具体接口与统计方法见：

- `docs/interfaces.md`
- `docs/adaptive_ber_method.md`
- `docs/c_equalizer_design.md`
- `docs/c_python_validation.md`

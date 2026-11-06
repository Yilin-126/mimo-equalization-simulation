# MIMO 均衡算法链路仿真与性能评估

> 项目类型：无线通信物理层算法仿真与工程验证  
> 主要语言：Python、C  
> 项目仓库：https://github.com/Yilin-126/mimo-equalization-simulation

## 1. 项目概述

本项目围绕 MIMO-OFDM 物理层接收与均衡算法，构建可复现的 Python 链路仿真环境，并实现小规模 C 语言 ZF/MMSE 均衡器。通过冻结 Python 基准向量、跨语言数值对齐、BER-SNR 自适应统计以及执行耗时测试，评估算法正确性与工程实现边界。

本项目参考 5G NR 物理层概念，属于**简化链路仿真**，并非完整的 3GPP NR 标准实现。

### 1.1 已完成功能

- QPSK、16QAM 调制及硬判决解调；OFDM IFFT/FFT、循环前缀、资源网格映射。
- AWGN、独立 Rayleigh MIMO 信道、频率平坦信道网格。
- 独立模块形式的 LS 导频信道估计与频域线性插值。
- Python ZF/MMSE 均衡，BER、RMS EVM、MSE 计算。
- **固定 2×2 信道、Perfect CSI** 条件下的 BER-SNR 扫描与自适应统计。
- C 语言 **1×1 与 2×2 方阵** ZF/MMSE 均衡；冻结参考向量和 C/Python 数值对齐。
- Python/C-ctypes 调用路径计时与独立的原生 C 循环计时。
- 分离的比特、噪声、信道随机流；固定信道 BER 的重复运行一致性验证。

### 1.2 尚未实现的内容

EPA/EVA 或其他多径信道、空间相关信道、主 BER 扫描的非理想 CSI 接入、C 批处理 OFDM 网格接口、4×4 C 均衡，以及不同信道模型之间共享基础随机样本的严格变量控制，**目前均未完成**。`random_control.enabled` 和 `reuse_realizations` 配置项尚未接入实际开关逻辑。

## 2. 项目目录说明

| 路径 | 作用 |
|---|---|
| `configs/simulation.json` | 仿真参数、信道系数及随机种子 |
| `src/mimo_sim/` | Python 收发、信道、均衡、指标、BER 停止及随机流模块 |
| `scripts/run_ber_scan.py` | 固定信道自适应 BER-SNR 扫描 |
| `scripts/plot_constellation.py` | QPSK 星座图 |
| `scripts/plot_adaptive_ber.py` | 自适应 BER 统计样本量可视化 |
| `scripts/plot_c_validation.py` | C/Python 数值误差可视化 |
| `scripts/check_c_equalizer.py` | 冻结基准的 C/Python 数值对齐 |
| `scripts/benchmark_c_python.py` | Python 与 C/ctypes 逐向量耗时比较 |
| `scripts/export_equalizer_vectors.py` | 冻结基准数据生成，仅在有意更新基准时运行 |
| `c_src/equalizer.c`、`c_src/equalizer.h` | C 1×1/2×2 ZF/MMSE 实现与接口 |
| `c_src/benchmark_core.c` | Windows 原生 C 循环耗时测试 |
| `tests/`、`tests/fixtures/` | 自动化测试及冻结参考 JSON |
| `results/` | CSV 与 PNG 结果（是否已提交以 Git 状态为准） |
| `docs/` | 设计、接口、环境及测试方法说明 |

## 3. 环境准备

已验证环境：Windows、Python 3.11、VS Code、Git、MSYS2 UCRT64 GCC 16.2.0。Python 依赖以 `requirements.txt` 为准。详细说明见 [`docs/development_environment.md`](docs/development_environment.md)。

在 **VS Code 的项目根目录** PowerShell 终端中执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

如果当前终端提示 `gcc` 无法识别，先执行：

```powershell
$env:Path = "C:\msys64\ucrt64\bin;" + $env:Path
gcc --version
```

当前 Python DLL 加载脚本对 MSYS2 UCRT64 依赖目录采用 `C:\msys64\ucrt64\bin` 路径。其他电脑如安装目录不同，应相应调整加载配置。

## 4. 仿真参数与随机控制

主配置文件：`configs/simulation.json`。当前 BER 实验采用 QPSK、2×2 MIMO、`S=14`、`K=1200`、`Nfft=2048`、`Ncp=144`；扫描 `0～20 dB`，间隔 `2 dB`。

单帧测试比特数：`14 × 1200 × 2 × 2 = 67,200`。每个 SNR 点优先在 ZF 与 MMSE **分别达到 200 个误码**后结束；否则最多运行 30 个完整帧，即 **2,016,000 比特**。

随机控制参数：

```json
{
  "bit_seed": 20261106,
  "noise_seed": 20261107,
  "channel_seed": 20261108
}
```

上述字段位于配置的 `random_control` 节点。当前 BER 扫描使用独立比特与噪声随机流；由于主信道是固定矩阵，未使用随机信道流。相同配置下两次运行的 CSV 内容已经验证一致。**该结果不能替代不同信道模型间的变量控制验证。**

主 BER 脚本以**时域接收信号平均功率**确定噪声方差。冻结 Python/C 参考向量的 SNR 和噪声方差定义另行记录，不应混淆。详见 [`docs/adaptive_ber_method.md`](docs/adaptive_ber_method.md)。

## 5. Python 测试与实验复现

在项目根目录运行：

```powershell
python -m pytest tests/ -q
python -m scripts.run_ber_scan
python -m scripts.plot_constellation
python -m scripts.plot_adaptive_ber
```

最近一次完整测试结果：**93 passed**。

输出包括：

- `results/ber_comparison.csv`：SNR、两种 BER、样本比特数、误码数及停止原因。
- `results/ber_comparison.png`：ZF/MMSE BER-SNR 曲线。
- `results/constellation_comparison.png`：均衡前后 QPSK 星座图。
- `results/adaptive_ber_statistics.png`：不同 SNR 下的样本量和停止原因。

高 SNR 下测得 BER 为零只表示当前样本内没有观测到误码。若近似假设误码事件独立，零误码时单侧 95% 置信上界约为 `3/N`；`N=2,016,000` 时约为 `1.49×10^-6`。

## 6. C 模块编译与数值一致性验证

C 接口：`zf_equalize_c()`、`mmse_equalize_c()`；信道矩阵按行优先存储的 `double complex`，仅支持 `n=1` 和 `n=2` 方阵。返回码：`0` 成功、`-1` 输入无效、`-2` 无法求解。`complex128` 与 C 复数内存的兼容性已在当前 GCC/NumPy 环境验证，其他平台需重新测试。

先编译 DLL，再运行验证：

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -shared c_src/equalizer.c -o c_src/equalizer.dll -lm
python -m scripts.check_c_equalizer
python -m scripts.plot_c_validation
```

冻结基准文件：`tests/fixtures/equalizer_reference.json`。包括 1×1 正常/弱信道、2×2 正常/近奇异信道，每类在 0 和 20 dB 下测量 ZF/MMSE，合计 **16 项对齐全部通过**；最大绝对误差 **`1.379e-10`**，阈值 **`1e-8`**。

**注意：** 不应为了让 C 测试通过而重新生成冻结参考数据。这些数据不构成任意病态信道的误差上界。

## 7. C/Python 性能量化

### 7.1 Python 与 C/ctypes 调用路径

```powershell
python -m scripts.benchmark_c_python
```

覆盖 1×1、2×2 两种天线配置，64/256/1200 个独立均衡向量以及 ZF/MMSE，合计 12 组。相同组的 Python 和 C 调用使用相同输入；预热后测量 5 次并取中位数。

| 算法 | 当前实测加速比范围 |
|---|---:|
| ZF | 6.68～8.47 倍 |
| MMSE | 2.95～3.57 倍 |

输出：`results/c_python_benchmark.csv` 与 `results/c_python_benchmark.png`。该结论比较的是**当前 Python 逐向量实现与 C/ctypes 逐向量调用路径**；包含接口处理开销，且两者求解实现不同，不代表 C 相对全部 NumPy 实现的普遍加速倍数。本轮12组对比的最大绝对误差约 `4.58e-15`。

### 7.2 原生 C 循环耗时

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic c_src/benchmark_core.c c_src/equalizer.c -o c_src/benchmark_core.exe -lm
.\c_src\benchmark_core.exe 2 1200 ZF
.\c_src\benchmark_core.exe 2 1200 MMSE
```

12组原生 C 测试统计如下：

| 天线配置 | ZF (ns/次) | MMSE (ns/次) |
|---|---:|---:|
| 1×1 | 16.115～16.313 | 26.376～27.834 |
| 2×2 | 51.948～52.922 | 85.808～87.397 |

完整数据：`results/c_core_benchmark.csv`。这项测试**不经过 Python/ctypes**，但仍含 C 循环和函数调用开销；与上一项计时采用不同的输入生成方式和计时流程，不能直接相减求得 ctypes 的纯开销。

`.exe` 和 `.dll` 为本地编译产物，原则上不作为源码交付。

## 8. 工程技术文档

| 文档 | 内容 |
|---|---|
| [`docs/design.md`](docs/design.md) | 链路方案、模块架构、已实现与计划边界 |
| [`docs/interfaces.md`](docs/interfaces.md) | Python/C 数据维度、接口、存储规则 |
| [`docs/c_equalizer_design.md`](docs/c_equalizer_design.md) | C 均衡实现、求解逻辑、误差与原生C计时 |
| [`docs/c_python_validation.md`](docs/c_python_validation.md) | 冻结向量验证、跨语言性能比较 |
| [`docs/adaptive_ber_method.md`](docs/adaptive_ber_method.md) | 自适应统计、独立随机流及结果解释 |
| [`docs/development_environment.md`](docs/development_environment.md) | 开发环境与依赖 |
| [`docs/theory_notes.md`](docs/theory_notes.md) | 相关理论笔记 |

新增模块、配置项或测试结果时，应与代码一起更新相关 `docs/*.md` 及本 README，避免文档描述落后于实际实现。

## 9. 实施进度与后续方向

**第一周：** 工程框架、配置文件、接口和开发环境。  
**第二周：** Python 调制/OFDM/信道/均衡、自动化测试、BER曲线及星座图。  
**第三周：** 冻结 Python 基准、C ZF/MMSE、16项数值对齐、自适应 BER、测试及技术文档。  
**补充工程基线：** 独立随机流、93项测试、固定信道复现、12组 Python/C-ctypes 性能数据、12组原生 C 计时以及对应文档。

后续计划：跨信道共同基础随机样本控制、空间相关与多径信道、非理想 CSI、C 批处理接口、4×4 扩展，以及严格配对的纯接口开销测试。以上均不应视为当前已完成成果。


## 多场景Rayleigh信道仿真与变量控制

### 1. 已实现的信道场景

当前支持两种2×2频率平坦Rayleigh信道：

| 场景 | 模型 | 参数 |
|---|---|---|
| independent | 独立Rayleigh信道 | 无空间相关 |
| rx_correlated_0p7 | 接收端相关Rayleigh信道 | ρ=0.7 |

信道模型位于 `src/mimo_sim/channel_scenarios.py`。

当前主BER比较采用Perfect CSI，尚未实现EPA/EVA多径和非理想CSI的性能比较。

### 2. 随机变量控制

随机数管理模块为 `src/mimo_sim/random_control.py`。

使用独立随机种子控制发送比特、基础噪声及基础信道。

配置文件 `configs/simulation.json` 提供：

- `reuse_realizations`：不同信道场景是否复用基础随机样本。
- `reuse_across_snr`：不同SNR点是否复用相同帧编号的基础随机样本。

正式受控对比实验中两个参数均设置为 `true`。

同一帧编号下，发送数据、基础噪声和基础信道随机样本保持一致。噪声幅度根据目标SNR调整。

### 3. 多场景BER统计

主脚本：

`scripts/run_scenario_ber.py`

运行命令：

```powershell
python -m scripts.run_scenario_ber
```

实验采用：

- 2×2 MIMO-OFDM
- QPSK
- Perfect CSI
- SNR范围0–20 dB，步长2 dB
- 每个SNR至少20帧
- 每帧67,200比特
- 每个SNR最大30帧

所有场景和算法使用相同的帧数量。

噪声方差由独立Rayleigh参考信道的接收功率计算，其他场景共用相同的噪声实现。因此图中的SNR代表参考信道目标接收SNR。

结果文件：

- `results/scenario_ber_comparison.csv`
- `results/scenario_ber_comparison.png`

当前实验观察到：在所测试条件下，相关Rayleigh信道的BER高于独立Rayleigh信道，MMSE通常优于ZF。

### 4. 自动化测试

运行：

```powershell
python -m pytest tests/ -q
```

最新验证结果：

**93 passed**

测试包含随机数复用开关、跨SNR复用、信道场景模型、多场景BER可复现性及集成验证。

### 5. 技术文档

- `docs/design.md`：系统设计及多场景信道方案
- `docs/interfaces.md`：模块接口和数据维度规范
- `docs/adaptive_ber_method.md`：BER统计与随机变量控制
- `docs/c_equalizer_design.md`：C均衡模块设计
- `docs/c_python_validation.md`：C/Python数值与性能验证

相关信道场景目前仅实现2×2接收端相关的简化平坦衰落模型。EPA/EVA多径、非理想CSI以及更一般的空间相关模型仍属于后续扩展。


## 10. 项目仓库

**GitHub：** https://github.com/Yilin-126/mimo-equalization-simulation

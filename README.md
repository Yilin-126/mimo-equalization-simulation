# MIMO 均衡算法链路仿真与性能评估

本项目构建了一套基于 Python 的 2×2 MIMO-OFDM 物理层链路仿真平台，并提供了 C 语言底层 ZF/MMSE 均衡模块的跨语言精度与性能验证。系统支持 QPSK/16QAM 调制、OFDM 收发、AWGN 信道、平坦/相关 Rayleigh 信道、三抽头频率选择性信道、LS 信道估计以及 BER/MSE 自动化统计。

## 1. 快速开始 (Quick Start)

### 1.1 环境与依赖
- **Python**: 3.11+
- **C 编译器**: MSYS2 UCRT64 GCC (Windows) 或 GCC (Linux/macOS)
- **依赖包**: 已严格锁定版本，详见 `requirements.txt` (`numpy==2.4.6`, `scipy==1.17.1`, `matplotlib==3.11.2`, `pytest==9.1.1`)

### 1.2 安装与核心测试
请在**项目根目录**下，于终端中依次执行：

```powershell
# 1. 创建并激活虚拟环境
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. 安装依赖
python -m pip install -r requirements.txt

# 3. 跑通全部核心测试用例（期望输出：105 passed）
python -m pytest tests/ -q
```

## 2. 典型场景复现命令

所有实验数据与图表均已归档至 `results/` 目录。各图表的原始 CSV 数据路径与复现命令详见 **[results/INDEX.md](results/INDEX.md)**。

### 2.1 核心 BER 与 MSE 仿真
```powershell
# 复现图7、图8：Perfect CSI vs LS CSI 以及 LS 估计 MSE
python -m scripts.run_csi_ber

# 复现图6：平坦信道 vs 三抽头多径信道
python -m scripts.run_multipath_ber

# 其他基础扫描与场景对比
python -m scripts.run_ber_scan
python -m scripts.run_scenario_ber
```

### 2.2 底层物理机理分析
```powershell
# 计算 LS 相对 Perfect 的 BER 恶化倍数 (支撑非理想 CSI 对 ZF/MMSE 影响的差异分析)
python -m scripts.analyze_ber_gap

# 统计空间相关性对 2x2 信道矩阵条件数及 ZF 噪声放大因子的影响
python -m scripts.analyze_condition_number
```

### 2.3 C 语言均衡模块验证与基准测试
如果在 PowerShell 中提示找不到 `gcc`，请切换至 MSYS2 UCRT64 终端执行。

```powershell
# 1. 编译 C 语言动态链接库
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -shared c_src/equalizer.c -o c_src/equalizer.dll -lm

# 2. 跨语言数值一致性验证（期望输出：16项通过，最大绝对误差 < 1e-10）
python -m scripts.check_c_equalizer

# 3. C 与 Python 性能对比测试（复现图3）
python -m scripts.benchmark_c_python

# 4. 原生 C 独立性能基准测试
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic c_src/benchmark_core.c c_src/equalizer.c -o c_src/benchmark_core.exe -lm
.\c_src\benchmark_core.exe 2 1200 ZF
```

## 3. 核心结论与图表预览

- **空间相关性与多径效应**：相关信道导致矩阵条件数变大，ZF 均衡噪声放大显著；三抽头多径引起的频率选择性衰落抬高了整体 BER，但 MMSE 凭借正则化项表现出更强的鲁棒性。
- **非理想 CSI 影响**：LS 估计误差导致 BER 恶化，ZF 因求逆放大误差恶化更为明显；MMSE 在 LS CSI 下依然保持比 ZF 更低的绝对误码率。
- **跨语言一致性**：C 模块与 Python 端参考实现的最大绝对误差为 `1.379e-10`；C 语言均衡模块在 Python 逐向量调用路径下，ZF 加速约 6.68-8.47 倍，MMSE 加速约 2.95-3.57 倍。

*(图表已保存至 results/ 目录，如 `csi_ber_comparison.png`、`multipath_ber_comparison.png` 等)*

## 4. 工程结构

```text
mimo_equalization/
├── configs/             # 统一参数配置文件 (simulation.json)
├── c_src/               # C 语言均衡器源码与编译产物
├── docs/                # 详细的设计、接口与验证文档
├── results/             # 实验输出的 CSV 原始数据与 PNG 图表
│   └── INDEX.md         # 实验结果索引清单（数据溯源）
├── scripts/             # 实验运行、性能基准与机理分析脚本
├── src/mimo_sim/        # 核心仿真模块（收发端、信道、均衡、估计）
├── tests/               # 105 项自动化单元与集成测试
├── README.md            # 项目说明文档
└── requirements.txt     # 锁定的环境依赖清单
```

## 5. 验证与文档

- **设计文档**：`docs/design.md`（系统架构）、`docs/interfaces.md`（模块接口）。
- **统计方法**：`docs/adaptive_ber_method.md`（共同随机样本与自适应停止控制）。
- **验证报告**：`docs/final_validation_report.md`（关键结果总结与物理机理分析）。

## 6. 已知局限 (Known Limitations)

1. **信道模型**：多径为自定义三抽头离散模型，非标准 EPA/EVA 信道模型。
2. **信道估计**：LS 估计采用理想正交训练序列，未考虑实际 5G NR DMRS 资源映射、插值与导频开销。
3. **算法边界**：C 语言均衡模块目前仅支持 1x1 与 2x2 复数方阵。
4. **系统范围**：本项目为物理层算法的仿真与性能验证平台，并非完整的 5G NR 协议栈实现。

**GitHub 仓库**: [https://github.com/Yilin-126/mimo-equalization-simulation](https://github.com/Yilin-126/mimo-equalization-simulation)
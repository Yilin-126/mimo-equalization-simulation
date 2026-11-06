# MIMO均衡算法链路仿真与性能评估

基于Python的2×2 MIMO-OFDM链路仿真与C语言ZF/MMSE均衡验证。项目包含QPSK/16QAM、OFDM、AWGN、平坦/相关Rayleigh、三抽头频率选择性信道、LS信道估计及BER评估。

## 1. 环境与安装

- Python 3.11；依赖版本见 `requirements.txt`
- C编译器：MSYS2 UCRT64 GCC；编译器配置见 `docs/development_environment.md`
- 在**项目根目录**执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m pytest tests/ -q
```

最近一次本地测试：**105 passed**（请以重新运行的实际结果为准）。

## 2. 运行实验

```powershell
python -m scripts.run_ber_scan
python -m scripts.run_scenario_ber
python -m scripts.run_multipath_ber
python -m scripts.run_csi_ber
```

运行后在 `results/` 查看对应的CSV和PNG。配置统一位于 `configs/simulation.json`。实验使用固定随机种子，支持跨场景、跨SNR基础样本复用。多场景对比使用共同样本预算。

C均衡器数值验证（需要先编译DLL）：

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -shared c_src/equalizer.c -o c_src/equalizer.dll -lm
python -m scripts.check_c_equalizer
```

原生C性能基准（Windows）：

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic c_src/benchmark_core.c c_src/equalizer.c -o c_src/benchmark_core.exe -lm
.\c_srcenchmark_core.exe 2 1200 ZF
```

## 3. 验证与文档

- Python自动化测试：105项通过；C/Python冻结参考向量：16项通过，最大绝对误差约 `1.379e-10`。
- 工程设计：`docs/design.md`；模块接口：`docs/interfaces.md`。
- BER与变量控制：`docs/adaptive_ber_method.md`。
- C模块设计：`docs/c_equalizer_design.md`；C/Python验证：`docs/c_python_validation.md`。
- 关键结果与局限：`docs/final_validation_report.md`。

**范围说明：** 多径为自定义三抽头模型，并非标准EPA/EVA；LS使用理想正交训练，不计NR DMRS开销；C模块仅支持1×1、2×2方阵。项目为算法仿真验证，不是完整5G NR协议实现。

仓库：https://github.com/Yilin-126/mimo-equalization-simulation

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
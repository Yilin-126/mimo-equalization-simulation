# MIMO-OFDM链路仿真方案设计

## 1. 项目目标

基于Python构建简化MIMO-OFDM链路仿真平台，
实现信道估计、ZF/MMSE均衡和性能评估。

## 2. 系统架构

发送端：
随机比特 → QPSK/16QAM调制 → 资源映射
→ OFDM调制(IFFT+CP)

信道：
AWGN / EPA / EVA多径信道

接收端：
OFDM解调(去CP+FFT) → DMRS信道估计
→ ZF/MMSE均衡 → 硬判决解调 → 性能统计

## 3. 仿真参数

| 参数 | 默认配置 |
|---|---|
| 子载波数量 | 1200 |
| OFDM符号数量 | 14 |
| 循环前缀 | Normal CP |
| 调制方式 | QPSK |
| 天线配置 | 2×2 MIMO |
| 信道模型 | AWGN |
| 均衡算法 | MMSE |
| SNR | 10 dB |
| SNR扫描范围 | 0–20 dB，步长2 dB |

所有参数统一保存在configs/simulation.json中。

## 4. 功能模块

| 模块 | 主要功能 |
|---|---|
| transmitter.py | 比特生成、数字调制、OFDM调制 |
| channel.py | 噪声添加、MIMO信道生成 |
| channel_estimation.py | LS信道估计、频域插值 |
| equalizer.py | ZF/MMSE均衡 |
| receiver.py | OFDM解调、硬判决 |
| metrics.py | BER、EVM、MSE计算 |
| main.py | 配置读取、仿真调度 |

模块间采用NumPy数组传递数据，
信号和信道矩阵使用复数类型。

## 5. 性能指标

### BER：误码率

BER = 错误比特数 / 总传输比特数

### EVM：误差向量幅度

EVM = sqrt(
    sum(|S_est - S_ref|²) / sum(|S_ref|²)
)

### 信道估计MSE

MSE = mean(|H_est - H_true|²)

### 运算耗时

使用高精度计时器统计均衡算法执行时间。

## 6. 工程规范

- 使用相对路径，不硬编码本地绝对路径。
- 仿真参数独立存放于JSON配置文件。
- 遵循PEP 8及Black格式规范。
- 使用Git和GitHub管理代码版本。
- 各模块独立开发与测试。

## 7. 开发计划

第一周：理论学习、方案设计、工程骨架搭建。

第二周：Python全链路仿真与ZF/MMSE实现。

第三周：C语言工程化、性能评估与项目交付。

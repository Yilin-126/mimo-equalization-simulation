# MIMO-OFDM 仿真模块接口规范

> 文档性质：接口协议｜范围：当前已实现的 Python 与 C 模块

## 1. 适用范围及基本约定

本规范说明当前实现中的数组维度、复数存储、函数输入输出与模块交互条件。项目属于简化物理层链路仿真，并非完整的 5G NR 标准协议实现。

- Python 数组使用 NumPy；复信号通常采用 `complex128`，比特采用 `int8`（取值 0 或 1）。
- `S`：OFDM 符号数；`K`：有效子载波数；`Nfft`：FFT 点数；`Ncp`：循环前缀采样点数。
- `Nt`：发射天线数；`Nr`：接收天线数；`B`：比特数。
- 资源网格维度顺序为**符号、子载波、天线**；完整频域信道网格维度为**符号、子载波、接收天线、发射天线**。
- 单个资源单元的信号模型为 `y = Hx + n`，其中 `H.shape=(Nr,Nt)`、`x.shape=(Nt,)`、`y.shape=(Nr,)`。
- Python ZF 支持 `Nr >= Nt` 的满列秩信道；当前 C 接口只支持 `n=1` 或 `n=2` 的方阵。

## 2. 关键数组维度

| 数据对象 | 维度 | 说明 |
|---|---|---|
| 原始比特 | `(B,)` | 长度需匹配调制位数 |
| 调制后的符号向量 | `(B / bits_per_symbol,)` | QPSK 或 16QAM |
| 有效发送资源网格 | `(S, K, Nt)` | 尚未补零 |
| IFFT 输入网格 | `(S, Nfft, Nt)` | 中心映射后使用 `ifftshift` |
| 发送时域信号（含 CP） | `(S, Nfft+Ncp, Nt)` | IFFT 和 CP 插入结果 |
| 接收时域信号（含 CP） | `(S, Nfft+Ncp, Nr)` | MIMO 信道及 AWGN 后 |
| 接收 FFT 网格 | `(S, Nfft, Nr)` | 提取有效频点前需 `fftshift` |
| 有效接收资源网格 | `(S, K, Nr)` | 中心频域切片 |
| 单资源单元信道 | `(Nr, Nt)` | 复数信道矩阵 |
| 平坦频域信道网格 | `(S, K, Nr, Nt)` | 同一矩阵广播，**不是多径模型** |
| 接收向量 | `(Nr,)` | Python 单向量均衡输入 |
| 发送估计向量 | `(Nt,)` | Python 单向量均衡输出 |
| LS 导频估计 | 与输入导频数组同形状 | 按元素计算 `Y/X` |
| 插值信道向量 | `(K,)` | 对实部和虚部分别线性插值 |

主 BER 扫描采用 **Perfect CSI**。导频 LS 估计和插值是独立模块，不能将其误写为已经集成到该主扫描中。

## 3. Python 模块函数约定

| 模块／函数 | 输入 | 输出及主要约束 |
|---|---|---|
| `transmitter.generate_bits` | `num_bits, rng` | `(num_bits,)` 二进制 `int8`，要求正整数长度 |
| `transmitter.modulate` | `bits, modulation_type` | QPSK/16QAM 归一化复符号向量 |
| `transmitter.map_to_resource_grid` | `(S,K,Nt), fft_size, K` | `(S,Nfft,Nt)`，资源映射后按 IFFT 顺序排列 |
| `transmitter.ofdm_modulate` | `(S,Nfft,Nt), cp_length` | `(S,Nfft+Ncp,Nt)` |
| `receiver.ofdm_demodulate` | `(S,Nfft+Ncp,Nr), cp_length, fft_size` | `(S,Nfft,Nr)` |
| `receiver.demodulate` | 1 维复符号、调制方式 | 扁平 `int8` 比特数组 |
| `channel.add_awgn` | 复信号、噪声方差、随机数生成器 | 与原信号同形状的含噪复数组 |
| `channel.generate_mimo_channel` | `Nt, Nr, rng` | `(Nr,Nt)` 独立 Rayleigh 系数 |
| `channel.generate_frequency_channel` | `S,K,Nt,Nr,rng` | `(S,K,Nr,Nt)` 平坦信道广播网格 |
| `channel_estimation.ls_channel_estimate` | 接收导频、参考导频 | 同形状 `Y/X`，参考导频不得含零 |
| `channel_estimation.interpolate_channel` | 导频估计、升序索引、`K` | `(K,)` 复数插值向量 |
| `equalizer.zf_equalize` | `y,H` | `(Nt,)`，要求适当维度且满列秩 |
| `equalizer.mmse_equalize` | `y,H,noise_variance` | `(Nt,)`，方差非负；假设单位发送符号协方差 |
| `metrics.calculate_ber` | 发送与接收比特 | `[0,1]` 内的浮点 BER |
| `metrics.calculate_evm` | 参考符号、估计符号 | 归一化 RMS EVM（线性比值） |
| `metrics.calculate_mse` | 真实、估计信道 | 非负浮点 MSE |
| `ber_stopping.get_stop_reason` | 两算法误码数、样本数及门限 | `target_reached`、`max_bits_reached` 或 `None` |
| `random_control.create_rngs` | 包含 `random_control` 的配置字典 | `bits`、`noise`、`channel` 三条独立 NumPy 随机流 |

输入合法性与异常行为以源码及对应单元测试为准。上述 `K` 在不同上下文中有可能表示**活动子载波数**或**独立均衡向量数量**；性能脚本中的 `K=64/256/1200` 是后者，不代表 C API 支持整块资源网格输入。

## 4. C语言接口与复数数据存储

声明文件：`c_src/equalizer.h`。

```c
int zf_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double complex *x
);

int mmse_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double noise_variance,
    double complex *x
);
```

| 参数 | 约定 |
|---|---|
| `h` | `(n,n)` 复数信道矩阵；行优先 `h[rx*n + tx]` |
| `y` | 长度为 `n` 的接收复数向量 |
| `n` | 只能为 1 或 2 |
| `noise_variance` | MMSE 使用，每个接收复样本的非负噪声方差 |
| `x` | 长度为 `n` 的复数输出缓冲区 |
| 返回码 | `0` 成功；`-1` 参数无效；`-2` 奇异或无法求解 |

C 使用 `double complex`；Python 桥接采用连续 NumPy `complex128` 数据，通过 `ctypes` 传递指针。此内存兼容性已经在**当前 Windows、MSYS2 GCC、NumPy 环境**验证；更换编译器或平台需重新验证 ABI。冻结 JSON 参考数据采用 `[real, imag]` 表示复数，与原始内存布局不是同一概念。

ZF 解线性系统 `H x_hat = y`，MMSE 解 `(H^H H + sigma^2 I)x_hat = H^H y`；当前 C 求解器采用复数高斯消元和部分主元选择。详情见 [`c_equalizer_design.md`](c_equalizer_design.md)。

## 5. 随机变量与对比实验约定

配置文件 `configs/simulation.json` 中的 `random_control` 分别包含 `bit_seed`、`noise_seed`、`channel_seed`。当前固定信道 BER 脚本只使用比特与噪声随机流，固定信道矩阵不消耗信道随机流。

同一帧的 ZF/MMSE 比较共用发送比特、信道和含噪接收数据。相同配置下两次主 BER 扫描已得到逐行相同的 CSV。

**尚未完成：** 不同信道模型的共同基础样本控制；配置项 `enabled`、`reuse_realizations` 对应的实际开关逻辑。未来多信道实验需预先定义一致的基础比特、噪声、信道随机样本及相同的 SNR/归一化口径，不能仅凭“种子相同”认定变量已控制。

## 6. 验证基线与文档关联

- Python 自动化测试：最新验收 **70 passed**。
- 冻结 C/Python 数值对齐：8 组场景 × 2 种算法 = **16 项全部通过**；最大绝对误差 `1.379e-10`，验收阈值 `1e-8`。
- Python/C-ctypes 调用路径计时：1×1、2×2，独立向量数 64/256/1200，ZF/MMSE 共 12 组。
- 原生 C 循环计时：另外 12 组；不经过 Python，但输入和计时方法与上一组不同。

相关说明：[`design.md`](design.md)、[`c_python_validation.md`](c_python_validation.md)、[`adaptive_ber_method.md`](adaptive_ber_method.md)。

## 7. 尚未实现的接口

当前基线**不包含** EPA/EVA 多径接口、空间相关信道场景统一对比、非理想 CSI 主 BER 流程、C 批量 OFDM 网格接口以及 4×4 C 均衡器。后续开发应先更新接口规范，再扩展实现与单元测试。

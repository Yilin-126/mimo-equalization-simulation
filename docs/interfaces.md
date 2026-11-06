# 模块接口规范

## 1. 数据与存储约定

Python使用 `np.complex128`，下标顺序如下。C模块使用C11 `double complex`，矩阵按行主序存储；`ctypes`传输连续的复数双精度数组。

| 对象 | 维度 | 说明 |
|---|---|---|
| 资源栅格 | `(S,K,Nt)` | OFDM符号、有效子载波、发送天线 |
| FFT资源网格 | `(S,Nfft,Nt)` | NumPy FFT顺序 |
| 时域发送 | `(S,Nfft+CP,Nt)` | 含CP |
| 时域接收 | `(S,Nfft+CP,Nr)` | 含CP |
| 单个平坦信道 | `(Nr,Nt)` | 接收天线 × 发送天线 |
| 多径冲激响应 | `(D+1,Nr,Nt)` | 延迟 × 接收 × 发送 |
| 频率响应 | `(Nfft,Nr,Nt)` | FFT频点 × 接收 × 发送 |
| 有效频点信道 | `(K,Nr,Nt)` | 与有效子载波顺序一致 |

`S`为OFDM符号数，`K`为有效子载波数，`Nt/Nr`为天线数。FFT频谱映射、`fftshift`顺序须与收发脚本一致。

## 2. 主要Python接口

| 接口 | 输入 → 输出 | 约束 |
|---|---|---|
| `generate_bits` / `modulate` | 比特 → QPSK/16QAM符号 | 二进制比特及整组长度 |
| `map_to_resource_grid` / `ofdm_modulate` | `(S,K,Nt)` → `(S,Nfft+CP,Nt)` | `0<K≤Nfft`，有效CP |
| `ofdm_demodulate` / `demodulate` | 时域接收 → 频域栅格；符号 → 比特 | QPSK、16QAM |
| `zf_equalize` / `mmse_equalize` | `y:(Nr,)`、`H:(Nr,Nt)` → `(Nt,)` | `Nr≥Nt`；MMSE还需噪声方差 |
| `create_frame_samples` | 配置、SNR索引、帧索引、形状 → `bits/noise/channel` | 支持场景及跨SNR复用 |
| `apply_channel_model` | `(Nr,Nt)` → `(Nr,Nt)` | 独立/2接收天线相关Rayleigh |
| `create_scenario_frame` | 配置与场景 → `bits/noise/base_channel/channel` | 与共享样本机制一致 |
| `create_multipath_taps` | `(L,Nr,Nt)`、整数延迟、功率dB → `(D+1,Nr,Nt)` | 不重复非负延迟 |
| `apply_multipath_channel` | 时域TX、抽头 → 时域RX | 块内线性卷积模型 |
| `multipath_frequency_response` | 多径抽头、FFT大小 → `(Nfft,Nr,Nt)` | `Nfft≥D+1` |
| `estimate_mimo_channel_ls` | `H_true`、导频噪声、方差、FFT大小 → `H_est` | 理想正交训练、逐频点LS |
| `equalize_grid` | `(S,K,Nr)`、`(K,Nr,Nt)` → ZF/MMSE `(S,K,Nt)` | 逐有效子载波计算 |

主要模块路径：`src/mimo_sim/`；`equalize_grid` 位于 `scripts/run_multipath_ber.py`。

## 3. C接口

`c_src/equalizer.h` 提供：

```c
int zf_equalize_c(const double complex *h, const double complex *y, int n, double complex *x);
int mmse_equalize_c(const double complex *h, const double complex *y, int n, double noise_variance, double complex *x);
```

仅支持 `n=1` 或 `n=2` 方阵，`h`按行主序展平。返回状态：`0`成功；`-1`参数无效；`-2`无法求解。MMSE使用 `(HᴴH+σ²I)x=Hᴴy`，假设发射符号协方差为单位阵。详细约定见 `docs/c_equalizer_design.md`。

## 4. 输出与测试

- `scripts/run_ber_scan.py`：`ber_comparison.csv/png`
- `scripts/run_scenario_ber.py`：`scenario_ber_comparison.csv/png`
- `scripts/run_multipath_ber.py`：`multipath_ber_comparison.csv/png`
- `scripts/run_csi_ber.py`：`csi_ber_comparison.csv/png`、`csi_mse.csv/png`

BER结果记录SNR、算法/场景、BER、比特数、误码数、停止原因。当前测试105项通过，完整结果见 `docs/final_validation_report.md`。

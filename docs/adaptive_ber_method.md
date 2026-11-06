# 自适应BER统计与随机变量控制

## 1. BER统计

`BER = 累计错误比特数 / 累计测试比特数`。同一组对比对象共用每帧输入及统计比特预算。

| 参数 | 当前设置 |
|---|---:|
| 每帧信息比特（2×2 QPSK） | 67,200 |
| 目标错误比特数（每条曲线） | 200 |
| 多场景/多径/CSI最少帧数 | 20 |
| 每SNR最大帧数 | 30 |
| 每SNR最大比特数 | 2,016,000 |

满足最少帧数且各比较对象都达到目标误码数时停止，否则到达最大预算停止。固定信道BER脚本采用其自身的自适应停止策略。零误码仅表示本次有限样本未观察到错误。

## 2. 随机变量控制

代码：`src/mimo_sim/random_control.py`，配置：`configs/simulation.json`。

- `bit_seed`、`noise_seed`、`channel_seed`：分别控制比特、基础噪声和基础信道。
- `reuse_realizations=true`：跨信道场景复用基础样本；关闭则按`scenario_id`区分随机流。
- `reuse_across_snr=true`：同一帧编号跨SNR复用基础样本；改变SNR时只缩放噪声幅度。
- CSI导频噪声独立使用 `pilot_seed`，不影响数据信号的随机流。

核心接口：`create_frame_samples(config, snr_index, frame_index, num_bits, noise_shape, channel_shape, scenario_id="default")`。

## 3. 公平比较规则

- ZF/MMSE：同一接收信号、真实信道及发送比特。
- 不同信道：共享基础随机样本及实际数据噪声；由参考信道接收功率计算噪声方差，因此各场景实际接收功率可不同。
- Perfect/LS CSI：共享真实信道、发送比特及数据噪声，均衡器使用真实/LS估计信道；LS导频训练采用独立噪声模型。
- 所有同组曲线共享帧数，不对其中一条曲线单独提前停止。

## 4. 验证与边界

跨场景/跨SNR复用、停止条件、数据重复运行及多场景/多径/CSI集成测试均已覆盖；最近一次全项目测试：**105 passed**。

实验结果见 `results/`，关键数值见 `docs/final_validation_report.md`。结果为有限随机样本下的估计，不等于理论误码率界限。

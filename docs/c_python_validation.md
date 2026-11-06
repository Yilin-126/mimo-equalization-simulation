# C/Python数值对齐与性能验证

## 1. 数值一致性

冻结参考数据：`tests/fixtures/equalizer_reference.json`。覆盖1×1/2×2、正常/弱/近奇异信道、0/20 dB以及ZF/MMSE，共8个场景、16次比较。

运行：

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -shared c_src/equalizer.c -o c_src/equalizer.dll -lm
python -m scripts.check_c_equalizer
```

结果：**16/16通过**；最大绝对误差`1.379e-10`；验收阈值`1e-8`。此结果不代表任意病态矩阵均可稳定求解。

## 2. 性能比较

测试天线规模1×1、2×2；独立均衡向量数`K=64/256/1200`；各自覆盖ZF/MMSE，共12组。

| 指标 | ZF | MMSE |
|---|---|---|
| Python与C/ctypes接口级加速 | 约6.68–8.47倍 | 约2.95–3.57倍 |
| 1×1原生C单次调用 | 约16 ns | 约26–28 ns |
| 2×2原生C单次调用 | 约52 ns | 约86–87 ns |

数据及图：`results/c_python_benchmark.csv/png`、`results/c_core_benchmark.csv`、`results/c_python_error_comparison.png`。

## 3. 解释边界

接口级测试包含Python与ctypes调用开销；原生C基准采用另一调用路径。两者测试条件不完全相同，**不能将其耗时差直接视为ctypes开销**。上述性能数值仅代表当前设备及编译参数。

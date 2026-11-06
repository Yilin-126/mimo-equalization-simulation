# C语言ZF/MMSE模块设计说明

## 1. 模块范围

文件：`c_src/equalizer.c`、`c_src/equalizer.h`。使用C11 `double complex` 实现1×1、2×2方阵ZF/MMSE均衡；不支持更大矩阵或`Nr>Nt`的矩形信道。

## 2. 接口和数据

```c
int zf_equalize_c(const double complex *h, const double complex *y, int n, double complex *x);
int mmse_equalize_c(const double complex *h, const double complex *y, int n, double noise_variance, double complex *x);
```

`h`：行主序 `(n,n)` 复数矩阵；`y`：`(n,)`；`x`：输出`(n,)`；`noise_variance`：非负复噪声功率。返回 `0` 成功、`-1` 无效输入、`-2` 无法求解。Python/ctypes使用连续 `complex128` 数组传递数据。

## 3. 求解逻辑及边界

- ZF：求解 `Hx=y`，使用带部分主元的复数高斯消元。
- MMSE：求解 `(HᴴH+σ²I)x=Hᴴy`，假设单位发射符号功率。
- 近奇异信道下误差可能放大；仅验证了冻结向量及当前测试范围，不能声称适用于任意病态矩阵。

## 4. 验证摘要

冻结Python参考向量8组、ZF/MMSE共16次比较全部通过；最大绝对误差`1.379e-10`，阈值`1e-8`。详细测试见 `docs/c_python_validation.md`。

原生C执行路径（GCC `-O2`，当前Windows设备）平均单次调用：1×1 ZF约16 ns、MMSE约26–28 ns；2×2 ZF约52 ns、MMSE约86–87 ns。结果数据：`results/c_core_benchmark.csv`。

原生计时包含C函数/循环开销，不包含Python/ctypes；不能直接与不同计时方案的数据相减得到精确接口开销。

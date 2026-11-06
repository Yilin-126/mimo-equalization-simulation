# C语言ZF/MMSE均衡模块设计与验证说明

## 1. 模块目标

本模块在Python MIMO均衡算法的基础上，实现C语言版本的ZF与MMSE均衡器，并通过固定Python参考向量验证计算结果。

当前模块主要用于小规模MIMO线性均衡算法的数值验证和性能比较，不包含完整OFDM链路处理。

## 2. 文件组成

- `c_src/equalizer.h`：C函数接口声明与参数说明
- `c_src/equalizer.c`：ZF/MMSE均衡及复数线性方程组求解
- `scripts/check_c_equalizer.py`：C/Python数值一致性检查
- `scripts/benchmark_c_python.py`：跨语言接口耗时对比
- `tests/fixtures/equalizer_reference.json`：冻结Python参考向量

## 3. 接口约定

### 3.1 ZF均衡接口

```c
int zf_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double complex *x
);
```

输入：

- `h`：复数信道矩阵，维度为 `(n, n)`
- `y`：接收复数向量，维度为 `(n,)`
- `n`：当前支持1或2
- `x`：输出复数向量缓冲区，维度为 `(n,)`

### 3.2 MMSE均衡接口

```c
int mmse_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double noise_variance,
    double complex *x
);
```

MMSE额外输入 `noise_variance`，表示每个接收天线上的复高斯噪声方差。

### 3.3 返回码

- `0`：计算成功
- `-1`：输入参数无效
- `-2`：线性方程组不可求解

## 4. 复数存储及数据维度

Python采用NumPy `complex128`，C语言使用 `double complex`。

复数数据由实部和虚部构成；本项目在Windows GCC环境下，通过连续的NumPy复数数组与C语言接口传递数据。

信道矩阵采用行优先存储：

`h[rx * n + tx]`

其中：

- `rx`：接收天线索引
- `tx`：发送天线索引
- `n`：天线数量

Python侧传入C函数前，将数据转换为连续的 `complex128` 数组。

该内存接口已在当前GCC与Python环境下通过数值对齐测试。跨编译器或跨平台使用时，需要重新验证复数类型的内存兼容性。

## 5. 矩阵求解逻辑

### 5.1 ZF

对于接收模型：

`y = Hx + n`

ZF均衡求解：

`Hx_hat = y`

当前C实现针对1×1和2×2方阵，使用复数线性方程组求解获得发送符号估计。

### 5.2 MMSE

MMSE使用：

`x_hat = (H^H H + sigma² I)^(-1) H^H y`

其中：

- `H^H`：信道矩阵的共轭转置
- `sigma²`：噪声方差
- `I`：单位矩阵

该形式假设发送符号协方差为单位矩阵，并且接收噪声为空间白噪声。

### 5.3 数值求解

当前C实现采用带部分主元选择的复数高斯消元方式求解线性方程组，避免显式计算完整逆矩阵。

近奇异信道可能放大数值误差，因此需要单独测试。

## 6. 数值一致性验证

测试数据先由Python版本生成并冻结，然后用于验证C语言实现。

覆盖：

- 1×1正常信道
- 1×1弱信道
- 2×2正常信道
- 2×2近奇异信道
- 0 dB和20 dB两种SNR条件
- ZF和MMSE两种算法

合计：

- 8组测试场景
- 16项C/Python比较

验收阈值：

`1e-8`

当前验证结果：

- 16项全部通过
- 最大绝对误差：`1.379e-10`

其中最大误差出现在2×2近奇异信道的ZF测试中。

该结论仅适用于当前冻结的测试向量，不能代表所有病态矩阵条件下的误差上界。

## 7. 耗时基准测试

通过 `scripts/benchmark_c_python.py` 比较Python函数与C/ctypes调用路径的执行时间。

测试配置：

| 天线配置 | 独立均衡向量数量 | 算法 |
|---|---|---|
| 1×1 | 64、256、1200 | ZF、MMSE |
| 2×2 | 64、256、1200 | ZF、MMSE |

每组使用相同的信道矩阵、接收数据和噪声条件。

测试采用预热及5次重复计时，统计中位数。

加速比定义：

`Speedup = Python_ms / C_ctypes_ms`

### 实测结果

- ZF加速比：约6.68–8.47倍
- MMSE加速比：约2.95–3.57倍
- 12组基准测试均通过数值一致性检查
- 本轮最大绝对数值误差：约 `4.58e-15`

结果文件：

- `results/c_python_benchmark.csv`
- `results/c_python_benchmark.png`

### 结果解释

该耗时结果包含Python调用C函数时的ctypes接口及数组处理开销。

此外，Python实现使用NumPy线性代数函数，而C实现针对小矩阵采用专门的求解过程。

因此，当前测试比较的是两条具体执行路径，不应直接解释为C语言相对于所有Python或NumPy实现的通用加速倍数。

当前子载波规模表示独立均衡向量的数量，并不代表C接口已经支持整块OFDM资源网格的批量计算。

## 8. 已知限制

1. 当前C接口仅支持1×1和2×2方阵。
2. 尚未实现原生C批量处理接口。
3. 尚未完成纯C计算核与接口开销分离计时。
4. 近奇异信道仅覆盖选定测试条件。
5. 尚未扩展至相关信道、多径信道及非理想CSI实验。
6. 跨平台部署时需重新验证编译器及复数内存布局。

## 9. 验证与复现

在项目根目录执行：

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -shared c_src/equalizer.c -o c_src/equalizer.dll -lm
```

```powershell
python -m scripts.check_c_equalizer
```

```powershell
python -m scripts.benchmark_c_python
```

以上步骤用于重新编译C动态库、验证数值一致性并生成性能对比结果。


## 10. 原生C计算路径耗时测试

为区分C算法执行耗时与Python/ctypes接口开销，新增独立的C语言性能测试程序：

`c_src/benchmark_core.c`

该程序直接调用C版本ZF和MMSE均衡函数，不经过Python解释器或ctypes接口。

测试采用GCC `-O2`优化编译，通过Windows高精度计时器测量执行时间。每组预热后重复测量5次，使用中位数计算平均单次调用耗时。

### 10.1 测试结果

| 天线配置 | K | ZF (ns/次) | MMSE (ns/次) |
|---|---:|---:|---:|
| 1×1 | 64 | 16.313 | 27.834 |
| 1×1 | 256 | 16.250 | 26.924 |
| 1×1 | 1200 | 16.115 | 26.376 |
| 2×2 | 64 | 52.922 | 87.236 |
| 2×2 | 256 | 52.455 | 85.808 |
| 2×2 | 1200 | 51.948 | 87.397 |

结果表明，在当前测试环境下，1×1配置的单次调用耗时低于2×2配置。MMSE的计算耗时高于ZF。

对于相同天线配置，不同K下的平均单次调用耗时接近。这与当前C接口逐个处理独立均衡向量的实现方式一致。

### 10.2 测试边界

本次测试包含C语言函数调用、循环及少量计时相关开销，不包含Python/ctypes调用开销。

原生C测试与之前的Python/ctypes基准测试未使用完全相同的接收向量和计时流程，因此两组结果应分别展示，不能直接将耗时差额全部解释为接口开销。

本测试仅覆盖1×1和2×2矩阵，不能推断更大MIMO配置的性能。

### 10.3 复现命令

在项目根目录编译：

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic c_src/benchmark_core.c c_src/equalizer.c -o c_src/benchmark_core.exe -lm
```

执行示例：

```powershell
.\c_src\benchmark_core.exe 2 1200 ZF
.\c_src\benchmark_core.exe 2 1200 MMSE
```

完整12组测试结果保存于：

`results/c_core_benchmark.csv`

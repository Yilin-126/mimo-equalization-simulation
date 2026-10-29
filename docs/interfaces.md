# Module Interface Specification

## 1. Data Conventions

- Array library: NumPy
- Complex dtype: complex128
- Grid layout: (symbol, subcarrier, antenna)
- Channel layout: (symbol, subcarrier, rx, tx)
- Bit dtype: int8
- Matrix convention: Y = HX + N

## 2. Array Shapes

| Variable | Shape |
|---|---|
| TX grid | (S, K, Nt) |
| Channel H | (S, K, Nr, Nt) |
| RX grid | (S, K, Nr) |
| Estimated H | (S, K, Nr, Nt) |
| Equalized grid | (S, K, Nt) |
| Single RE H | (Nr, Nt) |
| Single RE y | (Nr,) |

## 3. Module Responsibilities

- transmitter: Bits, modulation, resource mapping, OFDM
- channel: Noise and MIMO channel generation
- channel_estimation: Pilot-based LS estimation
- equalizer: ZF and MMSE detection
- receiver: OFDM demodulation and bit decisions
- metrics: BER, EVM and MSE

## 4. Validation Requirements

Each implemented module must have:
- Fixed-input numerical tests
- Input shape checks
- Output shape checks
- Reproducible random seeds where applicable

## 5. OFDM Design Decisions

The active subcarrier mapping, FFT size,
CP sample lengths and DMRS pattern must be
specified before OFDM implementation.
```

这个文件相当于整个项目的**接口协议**。后续开发时，我们会按照这些定义逐个验证。

## 五、开始实现第一个功能模块

接下来开发 `transmitter.py`。我们先实现随机比特生成及QPSK/16QAM调制，OFDM部分暂时保留接口。

将 `src/mimo_sim/transmitter.py` 替换为：

<CodeBlock language="python" editable>
"""MIMO-OFDM transmitter."""

import numpy as np


def generate_bits(
    num_bits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate reproducible random bits."""
    if num_bits <= 0:
        raise ValueError("num_bits must be positive")

    return rng.integers(0, 2, size=num_bits, dtype=np.int8)


def modulate(
    bits: np.ndarray,
    modulation_type: str,
) -> np.ndarray:
    """Map bits to normalized QPSK or 16QAM symbols."""
    bits = np.asarray(bits)

    bits_per_symbol = {
        "QPSK": 2,
        "16QAM": 4,
    }

    if modulation_type not in bits_per_symbol:
        raise ValueError("Unsupported modulation")

    n_bits = bits_per_symbol[modulation_type]

    if (
        bits.ndim != 1
        or len(bits) == 0
        or len(bits) % n_bits != 0
        or not np.all((bits == 0) | (bits == 1))
    ):
        raise ValueError("Invalid bit sequence")

    groups = bits.reshape(-1, n_bits).astype(np.int8)

    if modulation_type == "QPSK":
        real = 1 - 2 * groups[:, 0]
        imag = 1 - 2 * groups[:, 1]
        return (real + 1j * imag) / np.sqrt(2)

    real = (1 - 2 * groups[:, 0]) * (3 - 2 * groups[:, 1])
    imag = (1 - 2 * groups[:, 2]) * (3 - 2 * groups[:, 3])

    return (real + 1j * imag) / np.sqrt(10)


def ofdm_modulate(
    resource_grid: np.ndarray,
    cp_length: int,
) -> np.ndarray:
    """OFDM modulation interface."""
    raise NotImplementedError

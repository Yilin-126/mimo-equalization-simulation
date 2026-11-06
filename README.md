# MIMO Equalization Simulation

Python-based simulation and performance evaluation of MIMO-OFDM physical-layer equalization algorithms, with C implementations and numerical cross-validation.

The project is inspired by 5G NR physical-layer concepts. It is a simplified research and learning simulation, not a complete 3GPP-compliant 5G NR implementation.

## 1. Project Overview

The project covers:

- QPSK and 16QAM modulation and demodulation
- OFDM transmission and reception
- AWGN and Rayleigh MIMO channel models
- Least-squares (LS) pilot-based channel estimation
- Frequency-domain channel interpolation
- Zero-Forcing (ZF) and Minimum Mean Square Error (MMSE) equalization
- Bit Error Rate (BER), Error Vector Magnitude (EVM), and Mean Squared Error (MSE)
- Adaptive BER-SNR performance evaluation
- C implementations of ZF/MMSE equalizers
- C/Python numerical validation using frozen reference vectors

The main simulation uses a 2x2 MIMO-OFDM link with a fixed channel matrix and perfect channel state information (CSI) for the BER comparison.

## 2. Project Structure

- `configs/`: JSON simulation parameters
- `src/mimo_sim/`: Python signal-processing modules
- `scripts/`: Simulation, visualization, and validation scripts
- `c_src/`: C ZF/MMSE implementation and header files
- `tests/`: Python unit and integration tests
- `tests/fixtures/`: Frozen numerical reference vectors
- `docs/`: Development environment and interface documentation
- `results/`: Generated simulation data and figures
- `main.py`: Main Python entry point
- `requirements.txt`: Pinned Python package dependencies

Important files:

- `configs/simulation.json`: Simulation settings
- `src/mimo_sim/equalizer.py`: Python ZF/MMSE algorithms
- `src/mimo_sim/ber_stopping.py`: Adaptive BER stopping rules
- `c_src/equalizer.c`: C equalizer implementation
- `c_src/equalizer.h`: C function declarations
- `scripts/export_equalizer_vectors.py`: Reference-vector generation
- `scripts/check_c_equalizer.py`: C/Python numerical comparison
- `scripts/run_ber_scan.py`: BER-SNR simulation
- `scripts/plot_constellation.py`: Constellation visualization
- `tests/fixtures/equalizer_reference.json`: Frozen Python references

## 3. Development Environment

The project was developed and validated using:

- Windows
- Python 3.11
- Visual Studio Code
- Git and GitHub
- GCC 16.2.0 through MSYS2 UCRT64

Python package dependencies are pinned in `requirements.txt`.

Development software and environment information are documented separately in:

`docs/development_environment.md`

## 4. Installation

Run the following commands from the project root.

### 4.1 Create a virtual environment

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 4.2 Install Python dependencies

```powershell
python -m pip install -r requirements.txt
```

### 4.3 Check the environment

```powershell
python --version
python -m pytest --version
gcc --version
```

The GCC command requires the compiler to be available in the current terminal PATH.

## 5. Configuration

Simulation parameters are stored in:

`configs/simulation.json`

The configuration includes:

- FFT size and cyclic prefix length
- Number of OFDM symbols and active subcarriers
- Transmit and receive antenna counts
- Modulation scheme
- Random seed
- SNR range and step
- Channel coefficients
- Adaptive BER stopping parameters

Algorithm parameters are loaded from the configuration file rather than being embedded in the main simulation script.

Project files are located relative to the project root using Python's `pathlib`.

### Adaptive BER parameters

The performance configuration includes:

```json
{
  "target_errors": 200,
  "max_bits_per_snr": 2016000
}
```

These values define the error target per equalizer and the maximum tested bits per SNR point.

Other existing configuration fields, including the channel matrix, must be retained.

## 6. Python Simulation

### 6.1 Run all tests

```powershell
python -m pytest tests/ -q
```

The latest verified test run completed with:

`67 passed`

### 6.2 Generate BER-SNR results

```powershell
python -m scripts.run_ber_scan
```

Outputs:

- `results/ber_comparison.csv`
- `results/ber_comparison.png`

The simulation compares ZF and MMSE equalizers using the same transmitted bits, channel, and received noisy signals for each SNR point.

The BER comparison uses a fixed 2x2 channel matrix and perfect CSI.

### 6.3 Generate constellation plots

```powershell
python -m scripts.plot_constellation
```

Output:

`results/constellation_comparison.png`

The figure compares:

1. Received QPSK symbols before equalization
2. Symbols after ZF equalization
3. Symbols after MMSE equalization

## 7. Adaptive BER Evaluation

A fixed number of simulated frames may produce too few bit errors at high SNR, making BER estimates statistically unreliable.

This project uses two stopping conditions:

1. Both ZF and MMSE accumulate at least the configured target number of bit errors.
2. The maximum number of tested bits per SNR is reached, using complete simulation frames.

The first condition reached ends the simulation for that SNR.

Both equalizers use the same samples and therefore have the same tested-bit count at each SNR.

The current settings are:

- Target errors per equalizer: 200
- Maximum tested bits per SNR: 2,016,000
- Modulation in the current BER experiment: QPSK
- MIMO configuration: 2x2
- CSI: Perfect
- SNR range: 0 to 20 dB, in 2 dB steps

The CSV records:

- `SNR_dB`
- `ZF_BER`
- `MMSE_BER`
- `Bits`
- `ZF_Errors`
- `MMSE_Errors`
- `Stop_Reason`

The possible stopping labels are:

- `target_reached`
- `max_bits_reached`

When the maximum sample budget is reached before the error target, the estimate is sample-limited.

A measured BER of zero means that no errors were observed in the simulated bits. It does not establish that the true BER is zero.

For independent bit-error observations, the approximate one-sided 95% upper confidence bound after zero errors in N bits is:

`BER_upper ≈ 3 / N`

For N = 2,016,000 bits, this is approximately:

`1.49e-6`

## 8. C Equalizer Implementation

The C implementation is located in:

- `c_src/equalizer.c`
- `c_src/equalizer.h`

It implements:

- Zero-Forcing equalization
- MMSE equalization
- Complex-valued linear-system solving

The current C implementation supports square 1x1 and 2x2 channel matrices.

The channel matrix is stored in row-major order with shape:

`(Nr, Nt)`

Complex values use C99/C11 complex double precision, corresponding to NumPy `complex128`.

### 8.1 Compile the C library

On Windows, install the MSYS2 UCRT64 GCC toolchain.

In PowerShell, add GCC to the current terminal PATH if necessary:

```powershell
$env:Path = "C:\msys64\ucrt64\bin;" + $env:Path
```

Compile the shared library from the project root:

```powershell
gcc -std=c11 -O2 -Wall -Wextra -Wpedantic -shared c_src/equalizer.c -o c_src/equalizer.dll -lm
```

The generated DLL is a local build artifact and is not committed to Git.

The current Python DLL-loading script uses the MSYS2 UCRT64 installation path `C:\msys64\ucrt64\bin`. If MSYS2 is installed elsewhere, that loader path must be adjusted before validation.

### 8.2 C function behavior

The C API provides:

`zf_equalize_c(...)`

`mmse_equalize_c(...)`

Both functions accept the channel matrix, received vector, dimension, and output buffer. MMSE additionally accepts the noise variance.

The implementations return an integer status code:

- `0`: Success
- `-1`: Invalid arguments
- `-2`: Singular or unsolvable system

Detailed function declarations are available in `c_src/equalizer.h`.

## 9. Frozen Python Reference Vectors

Before comparing C results with Python, reference data was generated using the Python implementation and saved to:

`tests/fixtures/equalizer_reference.json`

The reference dataset contains eight cases covering:

- 1x1 normal channel
- 1x1 weak channel
- 2x2 normal channel
- 2x2 near-singular channel

Each channel condition is evaluated at:

- 0 dB SNR
- 20 dB SNR

The reference data stores complex values as:

`[real, imag]`

The frozen dataset includes channel matrices, received signals, noise variances, and expected ZF/MMSE outputs.

The reference values are intentionally frozen and should not be regenerated simply to make a failing C implementation pass.

The reference-generation script is:

```powershell
python -m scripts.export_equalizer_vectors
```

Run this only when intentionally establishing a new reference dataset. It is not required for ordinary validation.

## 10. C/Python Numerical Validation

Compile the C library first, then run:

```powershell
python -m scripts.check_c_equalizer
```

The script:

1. Loads the compiled C shared library.
2. Reads the frozen Python reference dataset.
3. Converts complex data into contiguous NumPy arrays.
4. Calls the C functions using `ctypes`.
5. Compares C results against the frozen Python outputs.
6. Checks the maximum absolute complex error.

Numerical acceptance tolerance:

`1e-8`

The latest verified validation result was:

- 8 frozen test cases
- 2 algorithms per case
- 16 C/Python comparisons passed
- Maximum absolute error: `1.379e-10`

Passing these reference cases demonstrates numerical agreement for the tested inputs. It does not establish robustness for all possible channel conditions.

## 11. Testing and Verification

Run all Python tests:

```powershell
python -m pytest tests/ -q
```

Run the C/Python reference comparison:

```powershell
python -m scripts.check_c_equalizer
```

Regenerate BER results:

```powershell
python -m scripts.run_ber_scan
```

Regenerate constellation plots:

```powershell
python -m scripts.plot_constellation
```

Run Python code formatting:

```powershell
python -m black src scripts tests main.py
```

Python functions include documentation of their purpose, arguments, return values, and relevant exceptions.

The interface conventions and data dimensions are described in:

`docs/interfaces.md`

## 12. Results and Limitations

### Observed results

In the current fixed-channel 2x2 MIMO-OFDM simulation, MMSE generally shows lower measured BER than ZF over the evaluated low-to-moderate SNR range.

At high SNR, some simulation points record zero errors within the maximum sample budget.

The generated CSV records the actual sample counts and error counts so that these results can be interpreted correctly.

### Limitations

- The BER experiment uses a fixed 2x2 channel rather than a large set of randomly generated channel realizations.
- Perfect CSI is used in the main ZF/MMSE BER comparison.
- The channel is simplified and does not reproduce all 5G NR propagation or protocol details.
- The current C equalizer supports 1x1 and 2x2 square matrices only.
- Near-singular numerical behavior is checked using selected frozen test vectors.
- High-SNR BER estimates may be limited by the maximum simulated bit count.
- The current Windows DLL loader assumes a particular MSYS2 installation path.

Future extensions could include correlated MIMO channels, multipath fading, imperfect CSI, larger antenna configurations, and further C implementation tests.

## 13. Development Progress

### Week 1 - Framework and Interfaces

Completed:

- Project structure
- Development environment
- Configuration file
- Relative project paths
- Modulation and channel fundamentals
- Initial algorithm interfaces
- GitHub repository setup

### Week 2 - Python MIMO-OFDM Simulation

Completed:

- QPSK and 16QAM modulation
- OFDM modulation and demodulation
- AWGN and Rayleigh channel modules
- LS channel estimation
- ZF and MMSE equalization
- BER, EVM, and MSE metrics
- Unit and integration tests
- BER-SNR comparison
- Constellation visualization

### Week 3 - C Implementation and Performance Validation

Completed:

- Frozen Python reference dataset
- 1x1 and 2x2 reference cases
- Normal, weak, and near-singular channel cases
- C ZF and MMSE equalizers
- GCC compilation
- C/Python numerical comparison
- Adaptive BER stopping mechanism
- Automated stopping-rule tests
- Function documentation improvements
- Reproducible validation commands

## 14. Repository

GitHub:

https://github.com/Yilin-126/mimo-equalization-simulation
# MIMO Equalization Simulation

Python-based simulation and performance evaluation of
5G NR MIMO-OFDM equalization algorithms.

## 1. Project Overview

This project focuses on:
- QPSK and 16QAM modulation
- OFDM transmission and reception
- AWGN and MIMO channel models
- LS channel estimation
- ZF and MMSE equalization
- BER, EVM and MSE evaluation

## 2. Project Structure

- `configs/`: Simulation parameters
- `src/mimo_sim/`: Python algorithm modules
- `tests/`: Unit tests
- `docs/`: Technical documentation
- `results/`: Simulation outputs
- `c_src/`: C implementation (planned)
- `main.py`: Main entry point

## 3. Development Environment

- Python 3.11
- VS Code
- Git

Python package dependencies are listed in
`requirements.txt`.

## 4. Installation

Create a virtual environment:

```powershell
python -m venv .venv
```

Install Python dependencies:

```powershell
python -m pip install -r requirements.txt
```

## 5. Configuration

Simulation parameters are stored in:

`configs/simulation.json`

Parameters are loaded separately from algorithm code.
No machine-specific absolute paths are required.

## 6. Usage

Run the current configuration validation:

```powershell
python main.py
```

Run unit tests:

```powershell
python -m pytest tests/ -v
```

Note: The complete MIMO-OFDM simulation is under
development. The current main program only validates
configuration loading.

## 7. Development Progress

- Week 1: Framework and interface definition completed
- Week 2: Python algorithms and unit tests in progress
- Week 3: C implementation and performance evaluation planned

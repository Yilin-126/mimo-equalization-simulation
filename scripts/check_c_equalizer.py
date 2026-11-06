"""Validate C ZF/MMSE equalizers against frozen Python references."""

import os
import ctypes
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DLL_PATH = ROOT / "c_src" / "equalizer.dll"
REFERENCE_PATH = ROOT / "tests" / "fixtures" / "equalizer_reference.json"

TOLERANCE = 1e-8


def decode_complex(values: list) -> np.ndarray:
    """Decode stored real/imag pairs.

    Args:
        values: JSON data with [real, imag] pairs.

    Returns:
        Contiguous NumPy complex128 array.
    """
    data = np.asarray(values, dtype=np.float64)
    result = data[..., 0] + 1j * data[..., 1]
    return np.ascontiguousarray(result, dtype=np.complex128)


def load_library() -> ctypes.CDLL:
    """Load the compiled C equalizer library.

    Returns:
        DLL with configured ZF/MMSE function signatures.

    Raises:
        FileNotFoundError: If the DLL has not been compiled.
    """
    if not DLL_PATH.exists():
        raise FileNotFoundError(f"Missing DLL: {DLL_PATH}")

    os.add_dll_directory(r"C:\msys64\ucrt64\bin")
    library = ctypes.CDLL(str(DLL_PATH))

    library.zf_equalize_c.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_void_p,
    ]
    library.zf_equalize_c.restype = ctypes.c_int

    library.mmse_equalize_c.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_double,
        ctypes.c_void_p,
    ]
    library.mmse_equalize_c.restype = ctypes.c_int

    return library


def run_c_equalizer(
    library: ctypes.CDLL,
    channel: np.ndarray,
    received: np.ndarray,
    method: str,
    noise_variance: float,
) -> np.ndarray:
    """Execute a C equalizer on complex128 arrays.

    Args:
        library: Loaded C dynamic library.
        channel: Row-major MIMO channel, shape (n, n).
        received: Received signal, shape (n,).
        method: Either ZF or MMSE.
        noise_variance: Complex AWGN variance per receive antenna.

    Returns:
        Estimated complex symbol vector, shape (n,).

    Raises:
        ValueError: If dimensions or method are invalid.
        RuntimeError: If the C function reports a failure.
    """
    h = np.ascontiguousarray(channel, dtype=np.complex128)
    y = np.ascontiguousarray(received, dtype=np.complex128)

    if h.ndim != 2 or h.shape[0] != h.shape[1]:
        raise ValueError("Expected square channel matrix")

    n = h.shape[0]

    if n not in (1, 2) or y.shape != (n,):
        raise ValueError("Expected 1x1 or 2x2 MIMO inputs")

    output = np.empty(n, dtype=np.complex128)

    h_ptr = ctypes.c_void_p(h.ctypes.data)
    y_ptr = ctypes.c_void_p(y.ctypes.data)
    x_ptr = ctypes.c_void_p(output.ctypes.data)

    if method == "ZF":
        status = library.zf_equalize_c(h_ptr, y_ptr, n, x_ptr)
    elif method == "MMSE":
        status = library.mmse_equalize_c(h_ptr, y_ptr, n, noise_variance, x_ptr)
    else:
        raise ValueError("Unknown equalization method")

    if status != 0:
        raise RuntimeError(f"C {method} equalizer failed: status={status}")

    return output


def main() -> None:
    """Compare all C results against the frozen reference dataset.

    Returns:
        None. Raises AssertionError when validation fails.
    """
    library = load_library()

    with REFERENCE_PATH.open(encoding="utf-8") as file:
        dataset = json.load(file)

    total = 0
    max_error = 0.0

    for case in dataset["cases"]:
        channel = decode_complex(case["channel"])
        received = decode_complex(case["received"])
        noise_variance = case["noise_variance"]

        for method, expected_key in [
            ("ZF", "zf_expected"),
            ("MMSE", "mmse_expected"),
        ]:
            expected = decode_complex(case[expected_key])

            actual = run_c_equalizer(
                library,
                channel,
                received,
                method,
                noise_variance,
            )

            error = float(np.max(np.abs(actual - expected)))
            max_error = max(max_error, error)

            passed = error <= TOLERANCE
            status = "PASS" if passed else "FAIL"

            print(
                f"{case['name']:28s} " f"{method:4s} | " f"error={error:.3e} | {status}"
            )

            if not passed:
                raise AssertionError(
                    f"{case['name']} {method}: " f"error {error} exceeds tolerance"
                )

            total += 1

    print(f"\n{total} C/Python comparisons passed")
    print(f"Maximum absolute error: {max_error:.3e}")


if __name__ == "__main__":
    main()

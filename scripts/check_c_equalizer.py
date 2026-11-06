"""Validate C ZF/MMSE equalizers against frozen Python references."""

import ctypes
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DLL_PATH = ROOT / "c_src" / "equalizer.dll"
REFERENCE_PATH = ROOT / "tests" / "fixtures" / "equalizer_reference.json"

TOLERANCE = 1e-8


def decode_complex(values: list) -> np.ndarray:
    """Decode complex numbers stored as real and imaginary pairs.

    Args:
        values: Nested JSON-compatible arrays containing
            [real, imag] pairs. Supports complex vectors
            and matrices.

    Returns:
        Contiguous NumPy complex128 array with the same
        leading dimensions as the input data.
    """
    data = np.asarray(values, dtype=np.float64)
    result = data[..., 0] + 1j * data[..., 1]
    return np.ascontiguousarray(result, dtype=np.complex128)


def load_library() -> ctypes.CDLL:
    """Load the compiled C equalizer dynamic library.

    Configure ctypes argument and return types for the
    ZF and MMSE equalizer functions.

    Args:
        None. The DLL path is defined by DLL_PATH
        relative to the project root.

    Returns:
        Loaded ctypes.CDLL object with configured
        zf_equalize_c and mmse_equalize_c signatures.

    Raises:
        FileNotFoundError: If equalizer.dll does not exist.
        OSError: If the DLL or its dependencies cannot
            be loaded.
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
    """Execute the C ZF or MMSE equalizer using complex128 data.

    Python complex128 arrays are passed to C through
    contiguous memory pointers. The C implementation
    supports square 1x1 and 2x2 MIMO channel matrices.

    Args:
        library: Loaded C dynamic library containing
            the equalizer functions.
        channel: Complex MIMO channel matrix with
            shape (n, n), stored in row-major order.
        received: Received complex signal vector
            with shape (n,).
        method: Equalization method, either "ZF"
            or "MMSE".
        noise_variance: Nonnegative complex AWGN
            variance per receive antenna, used by MMSE.

    Returns:
        Estimated complex128 transmitted symbol vector
        with shape (n,).

    Raises:
        ValueError: If matrix dimensions are invalid,
            n is not 1 or 2, or method is unsupported.
        RuntimeError: If the C equalizer returns a
            nonzero error status.
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
    """Validate C equalizers against frozen Python test vectors.

    Load the reference JSON dataset and compare C ZF/MMSE
    outputs with Python reference results.

    Calculate the maximum absolute complex error for
    each comparison and check it against TOLERANCE.

    Args:
        None. Reference data is loaded from
        tests/fixtures/equalizer_reference.json.

    Returns:
        None. Prints PASS or FAIL for each comparison,
        the number of successful comparisons, and
        the maximum absolute numerical error.

    Raises:
        FileNotFoundError: If the DLL or reference file
            does not exist.
        RuntimeError: If a C equalizer reports failure.
        AssertionError: If any numerical comparison
            exceeds the specified tolerance.
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

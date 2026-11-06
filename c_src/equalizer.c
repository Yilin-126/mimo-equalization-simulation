#include "equalizer.h"

#include <complex.h>
#include <math.h>
#include <stddef.h>

/**
 * Solve a complex linear system of size 1 or 2.
 *
 * Args:
 *   a: Row-major coefficient matrix (n x n).
 *   b: Right-hand-side vector (n).
 *   n: Matrix dimension (1 or 2).
 *   x: Output solution vector (n).
 *
 * Returns:
 *   0 on success, -1 for invalid inputs,
 *   -2 for singular or numerically degenerate matrices.
 */
static int solve_system(
    const double complex *a,
    const double complex *b,
    int n,
    double complex *x
) {
    if (!a || !b || !x || (n != 1 && n != 2)) {
        return -1;
    }

    if (n == 1) {
        if (cabs(a[0]) == 0.0) {
            return -2;
        }

        x[0] = b[0] / a[0];
        return 0;
    }

    double complex m[4] = {a[0], a[1], a[2], a[3]};
    double complex v[2] = {b[0], b[1]};

    if (cabs(m[2]) > cabs(m[0])) {
        double complex temp = m[0];
        m[0] = m[2];
        m[2] = temp;

        temp = m[1];
        m[1] = m[3];
        m[3] = temp;

        temp = v[0];
        v[0] = v[1];
        v[1] = temp;
    }

    if (cabs(m[0]) == 0.0) {
        return -2;
    }

    double complex factor = m[2] / m[0];
    m[3] -= factor * m[1];
    v[1] -= factor * v[0];

    if (cabs(m[3]) == 0.0) {
        return -2;
    }

    x[1] = v[1] / m[3];
    x[0] = (v[0] - m[1] * x[1]) / m[0];

    return 0;
}

/**
 * Perform Zero-Forcing MIMO equalization.
 *
 * Args:
 *   h: Complex MIMO channel matrix (n x n), row-major.
 *   y: Received signal vector (n).
 *   n: Number of antennas, 1 or 2.
 *   x: Equalized output vector (n).
 *
 * Returns:
 *   0 on success or a negative error code.
 */
int zf_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double complex *x
) {
    return solve_system(h, y, n, x);
}

/**
 * Perform MMSE MIMO equalization.
 *
 * Args:
 *   h: Complex MIMO channel matrix (n x n), row-major.
 *   y: Received signal vector (n).
 *   n: Number of antennas, 1 or 2.
 *   noise_variance: Nonnegative complex noise power.
 *   x: Equalized output vector (n).
 *
 * Returns:
 *   0 on success or a negative error code.
 */
int mmse_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double noise_variance,
    double complex *x
) {
    if (!h || !y || !x || (n != 1 && n != 2)
        || !isfinite(noise_variance) || noise_variance < 0.0) {
        return -1;
    }

    if (noise_variance == 0.0) {
        return zf_equalize_c(h, y, n, x);
    }

    double complex gram[4] = {0};
    double complex rhs[2] = {0};

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            for (int k = 0; k < n; ++k) {
                gram[i * n + j] +=
                    conj(h[k * n + i]) * h[k * n + j];
            }
        }

        gram[i * n + i] += noise_variance;

        for (int k = 0; k < n; ++k) {
            rhs[i] += conj(h[k * n + i]) * y[k];
        }
    }

    return solve_system(gram, rhs, n, x);
}

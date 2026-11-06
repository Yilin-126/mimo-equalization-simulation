#ifndef MIMO_EQUALIZER_H
#define MIMO_EQUALIZER_H

#include <complex.h>

/**
 * Perform ZF equalization.
 *
 * Args:
 *   h: Complex channel matrix, row-major (n x n).
 *   y: Received signal vector (n).
 *   n: Number of antennas (1 or 2).
 *   x: Equalized output vector (n).
 *
 * Returns:
 *   0 on success.
 *  -1 for invalid input.
 *  -2 for singular matrix.
 */
int zf_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double complex *x
);

/**
 * Perform MMSE equalization.
 *
 * Args:
 *   h: Complex channel matrix, row-major (n x n).
 *   y: Received signal vector (n).
 *   n: Number of antennas (1 or 2).
 *   noise_variance: Nonnegative complex noise variance.
 *   x: Equalized output vector (n).
 *
 * Returns:
 *   0 on success.
 *  -1 for invalid input.
 *  -2 if the system cannot be solved.
 */
int mmse_equalize_c(
    const double complex *h,
    const double complex *y,
    int n,
    double noise_variance,
    double complex *x
);

#endif

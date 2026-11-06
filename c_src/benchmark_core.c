/* Benchmark the native C ZF/MMSE computation path on Windows. */

#include "equalizer.h"

#include <complex.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>

#define REPEATS 5
#define NOISE_VARIANCE 0.1

static volatile double result_sink = 0.0;

/**
 * Read a high-resolution timestamp.
 *
 * Args:
 *     None.
 *
 * Returns:
 *     Performance-counter ticks.
 */
static LARGE_INTEGER counter_now(void)
{
    LARGE_INTEGER value;
    QueryPerformanceCounter(&value);
    return value;
}

/**
 * Compare two elapsed-time values for sorting.
 *
 * Args:
 *     a: Pointer to the first double value.
 *     b: Pointer to the second double value.
 *
 * Returns:
 *     Negative, zero, or positive comparison result.
 */
static int compare_double(const void *a, const void *b)
{
    double x = *(const double *)a;
    double y = *(const double *)b;
    return (x > y) - (x < y);
}

/**
 * Run a native C equalizer timing experiment.
 *
 * Args:
 *     argc: Number of command-line arguments.
 *     argv: Antenna count, vector count, and method.
 *
 * Returns:
 *     Zero on success, nonzero on invalid input or C failure.
 */
int main(int argc, char **argv)
{
    if (argc != 4) {
        fprintf(stderr, "Usage: benchmark_core.exe N K ZF|MMSE\n");
        return 1;
    }

    int n = atoi(argv[1]);
    int k = atoi(argv[2]);
    int method = -1;

    if (argv[3][0] == 'Z' && argv[3][1] == 'F'
        && argv[3][2] == '\0') {
        method = 0;
    } else if (argv[3][0] == 'M'
               && argv[3][1] == 'M'
               && argv[3][2] == 'S'
               && argv[3][3] == 'E'
               && argv[3][4] == '\0') {
        method = 1;
    }

    if ((n != 1 && n != 2) || k <= 0 || method < 0) {
        fprintf(stderr, "Invalid benchmark arguments\n");
        return 1;
    }

    double complex h[4] = {
        1.0 + 0.2 * I,
        0.3 - 0.1 * I,
        0.2 + 0.1 * I,
        0.9 - 0.2 * I
    };

    if (n == 1) {
        h[0] = 1.0 + 0.2 * I;
    }

    double complex *received =
        malloc((size_t)k * (size_t)n * sizeof(double complex));

    if (received == NULL) {
        fprintf(stderr, "Memory allocation failed\n");
        return 1;
    }

    for (int j = 0; j < k; ++j) {
        for (int antenna = 0; antenna < n; ++antenna) {
            double t = (double)(j + 1 + antenna * 17);
            received[(size_t)j * n + antenna] =
                cos(t * 0.13) + I * sin(t * 0.17);
        }
    }

    LARGE_INTEGER frequency;
    if (!QueryPerformanceFrequency(&frequency)) {
        free(received);
        fprintf(stderr, "High-resolution timer unavailable\n");
        return 1;
    }

    int batches = 200000 / k;
    if (batches < 1) {
        batches = 1;
    }

    double times[REPEATS];
    double complex output[2];

    /* Preheat the native equalizer before measuring. */
    for (int j = 0; j < k; ++j) {
        const double complex *y = received + (size_t)j * n;

        int status = method == 0
            ? zf_equalize_c(h, y, n, output)
            : mmse_equalize_c(
                  h, y, n, NOISE_VARIANCE, output);

        if (status != 0) {
            free(received);
            fprintf(stderr, "Equalizer warmup failed\n");
            return 1;
        }
    }

    for (int repeat = 0; repeat < REPEATS; ++repeat) {
        LARGE_INTEGER start = counter_now();

        for (int batch = 0; batch < batches; ++batch) {
            for (int j = 0; j < k; ++j) {
                const double complex *y =
                    received + (size_t)j * n;

                int status = method == 0
                    ? zf_equalize_c(h, y, n, output)
                    : mmse_equalize_c(
                          h, y, n, NOISE_VARIANCE, output);

                if (status != 0) {
                    free(received);
                    fprintf(stderr, "Equalizer failed\n");
                    return 1;
                }

                if (j == k - 1) {
                    result_sink += creal(output[0]);
                }
            }
        }

        LARGE_INTEGER end = counter_now();

        double seconds =
            (double)(end.QuadPart - start.QuadPart)
            / (double)frequency.QuadPart;

        times[repeat] = seconds * 1000.0;
    }

    qsort(
        times,
        REPEATS,
        sizeof(double),
        compare_double
    );

    double median_ms = times[REPEATS / 2];
    double calls = (double)batches * (double)k;
    double ns_per_call = median_ms * 1e6 / calls;

    printf(
        "N=%d K=%d method=%s calls=%.0f "
        "median_ms=%.6f ns_per_call=%.3f\n",
        n,
        k,
        method == 0 ? "ZF" : "MMSE",
        calls,
        median_ms,
        ns_per_call
    );

    free(received);
    return 0;
}

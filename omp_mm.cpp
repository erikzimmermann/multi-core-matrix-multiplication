#include "omp_mm.h"
#include <omp.h>
#include <iostream>
#include <math.h>

void multiplyMatrixPart(float *a, float *b, float *c, int n, int a_row, int b_col, int block_size) {
    int remaining = std::min(n - a_row, block_size);

    for (int i = 0; i < remaining; i++) {
        for (int j = 0; j < remaining; j++) {
            float *cPtr = c + (a_row + i) * n + b_col + j;
            float *aPtr = a + (a_row + i) * n;
            float *bPtr = b + b_col + j;

            for (int k = 0; k < n; ++k) {
                *cPtr += *(aPtr++) * *bPtr;
                bPtr += n;
            }
        }
    }
}

void multiplyMatrixCannon(float *a, float *b, float *c, int block_size, int n) {
    int iter = std::ceil(float(n) / float(block_size));

    #pragma omp parallel default(none) shared(a, b, c, n, iter, block_size)
    #pragma omp single
    {
        for (int i = 0; i < iter; ++i) {
            for (int j = 0; j < iter; ++j) {
                #pragma omp task
                multiplyMatrixPart(a, b, c, n, i * block_size, j * block_size, block_size);
            }
        }
    }
}

void multiplyMatrixOMP(float *a, float *b, float *c, int n, int threads) {
    if (threads > 0) omp_set_num_threads(threads);
    else omp_set_num_threads(omp_get_max_threads());

    int block_size = 32;
    if (n % block_size != 0) {
        block_size = 25;
        if (n % block_size != 0) {
            std::cerr << "Invalid block size. n=" << n << ", block_size=" << block_size << std::endl;
            return;
        }
    }

    multiplyMatrixCannon(a, b, c, block_size, n);
}

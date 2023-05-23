#include "omp_mm.h"
#include <omp.h>
#include <math.h>

//const int block_size = 32;

void multiplyMatrixPart(float *a, float *b, float *c, int n, int a_row, int b_col, int block_size) {
    int row_remaining = std::min(n - a_row, block_size);
    int col_remaining = std::min(n - b_col, block_size);

    for (int i = 0; i < row_remaining; i++) {
        for (int j = 0; j < col_remaining; j++) {
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

void multiplyMatrixCannon(float *a, float *b, float *c, int n, int block_size) {
    int iter = (n + block_size - 1) / block_size;

    #pragma omp parallel for collapse(2)
    for (int i = 0; i < iter; i++) {
        for (int j = 0; j < iter; j++) {
            multiplyMatrixPart(a, b, c, n, i * block_size, j * block_size, block_size);
        }
    }
}

void multiplyMatrixOMP(float *a, float *b, float *c, int n, int threads) {
    if (threads > 0) omp_set_num_threads(threads);
    else omp_set_num_threads(omp_get_max_threads());

    auto sizes = new int[10] { 1, 2, 4, 8, 16, 32, 64, 128, 256, 512 };

    for (int i = 0; i < 10; ++i) {
        multiplyMatrixCannon(a, b, c, n, sizes[i]);
    }
}

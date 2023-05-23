#include "omp_mm.h"
#include <omp.h>
#include <iostream>
#include <math.h>

void multiplyMatrixBlock(float *a, float *b, float *c,
                       int crow, int ccol,
                       int arow, int acol,
                       int brow, int bcol,
                       int l, int m, int n, int N) {
    int lhalf[3], mhalf[3], nhalf[3];

    if (m * n > 32768) {
        lhalf[0] = 0; lhalf[1] = l/2; lhalf[2] = l - l/2;
        mhalf[0] = 0; mhalf[1] = m/2; mhalf[2] = l - m/2;
        nhalf[0] = 0; nhalf[1] = n/2; nhalf[2] = l - n/2;

        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                for (int k = 0; k < 2; k++) {
                    multiplyMatrixBlock(a, b, c,
                                      crow + lhalf[i], ccol + mhalf[j],
                                      arow + lhalf[i], acol + mhalf[k],
                                      brow + mhalf[k], bcol + nhalf[j],
                                      lhalf[i + 1], mhalf[k + 1], nhalf[k + 1], N);
                }
            }
        }
    } else {
        #pragma omp parallel
        {
            #pragma omp for
            for (int i = 0; i < l; i++) {
                for (int j = 0; j < n; j++) {
                    float *aptr = a + (arow + i) * N + acol;
                    float *bptr = b + brow * N + bcol + j;
                    float *cptr = c + (crow + i) * N + (ccol + j);

                    float temp = 0.0f;
                    #pragma omp simd reduction(+:temp)
                    for (int k = 0; k < m; k++) {
                        temp += *(aptr++) * *bptr;
                        bptr += N;
                    }

                    #pragma omp critical
                    {
                        *cptr += temp;
                    }
                }
            }
        }
    }
}

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

void multiplyMatrixCannon(float *a, float *b, float *c, int n) {
    int block_size = 32;
    if (n % block_size != 0) {
        block_size = 25;
        if (n % block_size != 0) {
            std::cerr << "Invalid block size. n=" << n << ", block_size=" << block_size << std::endl;
            return;
        }
    }

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

//    multiplyMatrixBlock(a, b, c, 0, 0, 0, 0, 0, 0, n, n, n, n);
    multiplyMatrixCannon(a, b, c, n);
}

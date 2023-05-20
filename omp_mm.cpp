#include "omp_mm.h"
#include <omp.h>

void multiplyMatrixOMP(float *a, float *b, float *c,
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
                    multiplyMatrixOMP(a, b, c,
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

void multiplyMatrixOMP(float *a, float *b, float *c, int n) {
    multiplyMatrixOMP(a, b, c, 0, 0, 0, 0, 0, 0, n, n, n, n);
}

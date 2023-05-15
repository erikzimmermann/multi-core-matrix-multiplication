#include "mm.h"

void multiplyMatrix(float *a, float *b, float *c,
                    int crow, int ccol,
                    int arow, int acol,
                    int brow, int bcol,
                    int l, int m, int n, int N) {
    int lhalf[3], mhalf[3], nhalf[3];
    int i, j, k;
    float *aptr, *bptr, *cptr;

    if (m * n > THRESHOLD) {
        lhalf[0] = 0; lhalf[1] = l/2; lhalf[2] = l - l/2;
        mhalf[0] = 0; mhalf[1] = m/2; mhalf[2] = l - m/2;
        nhalf[0] = 0; nhalf[1] = n/2; nhalf[2] = l - n/2;

        for (i = 0; i < 2; i++) {
            for (j = 0; j < 2; j++) {
                for (k = 0; k < 2; k++) {
                    multiplyMatrix(a, b, c,
                                   crow + lhalf[i], ccol + mhalf[j],
                                  arow + lhalf[i], acol + mhalf[k],
                                  brow + mhalf[k], bcol + nhalf[j],
                                  lhalf[i + 1], mhalf[k + 1], nhalf[k + 1], N);
                }
            }
        }
    } else {
        for (i = 0; i < l; i++) {
            for (j = 0; j < n; j++) {
                cptr = c + (crow + i) * N + (ccol + j);
                aptr = a + (arow + i) * N + acol;
                bptr = b + brow * N + bcol + j;

                for (k = 0; k < m; k++) {
                    *cptr += *(aptr++) * *bptr;
                    bptr += N;
                }
            }
        }
    }
}

void multiplyMatrix(float *a, float *b, float *c, int n) {
    multiplyMatrix(a, b, c, 0, 0, 0, 0, 0, 0, n, n, n, n);
}

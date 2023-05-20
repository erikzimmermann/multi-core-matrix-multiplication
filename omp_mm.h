#ifndef MATRIX_MULTIPLICATION_WITH_OPENMP_MM_H
#define MATRIX_MULTIPLICATION_WITH_OPENMP_MM_H

const int THRESHOLD = 32768;

void multiplyMatrixOMP(float *a, float *b, float *c, int size);

#endif //MATRIX_MULTIPLICATION_WITH_OPENMP_MM_H

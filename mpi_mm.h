#ifndef MATRIX_MULTIPLICATION_WITH_OPENMP_MPI_MM_H
#define MATRIX_MULTIPLICATION_WITH_OPENMP_MPI_MM_H

#include <mpi.h>

void distributeMatrix(const float *a, const float *b, int N);

void collectMatrix(float *c, int N);

void receiveMatrixPart(float *a, float *b, int block_size);

void handleMatrixPart(int N);


#endif //MATRIX_MULTIPLICATION_WITH_OPENMP_MPI_MM_H

#ifndef MATRIX_MULTIPLICATION_WITH_OPENMP_MPI_MM_H
#define MATRIX_MULTIPLICATION_WITH_OPENMP_MPI_MM_H

#include <mpi.h>

void distributeMatrix(double *a, double *b, int N);

void collectMatrix(double *c, int N);

void receiveMatrixPart(double *a, double *b, int block_size);

void handleMatrixPart(int N);


#endif //MATRIX_MULTIPLICATION_WITH_OPENMP_MPI_MM_H

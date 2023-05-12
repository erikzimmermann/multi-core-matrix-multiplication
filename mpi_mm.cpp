#include "mpi_mm.h"
#include <mpi.h>
#include <random>

void distributeMatrix(double *a, double *b, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int blockSize = N / int(log2(processes - 1));

    // start at 1 because 0 is the main process
    for (int i = 1; i < processes; ++i) {
        // build buffer with square matrix part A
        double buffer[blockSize][blockSize];

        for(int j = 0; j < blockSize; ++j) {
            int row = j + i * blockSize;
            for (int k = 0; k < blockSize; ++k) {
                int col = k + i * blockSize;
                buffer[j][k] = *(a + row * N + col);
                a++;
            }
        }

        // tag=0 for matrix a
        MPI_Send(&buffer[0][0], blockSize * blockSize, MPI_DOUBLE, i, 0, MPI_COMM_WORLD);

        // build buffer with square matrix part B
        for(int j = 0; j < blockSize; ++j) {
            int row = j + i * blockSize;
            for (int k = 0; k < blockSize; ++k) {
                int col = k + i * blockSize;
                buffer[j][k] = *(a + row * N + col);
                b++;
            }
        }

        // tag=1 for matrix b
        MPI_Send(&buffer[0][0], blockSize * blockSize, MPI_DOUBLE, i, 1, MPI_COMM_WORLD);
    }
}

void receiveMatrixPart(double *a, double *b, int block_size) {
    // build buffer with square matrix part
    double buffer[block_size][block_size];

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // tag=0 for matrix a
    MPI_Recv(&buffer[0][0], block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for(int j = 0; j < block_size; ++j) {
        for (int k = 0; k < block_size; ++k) {
            *a = buffer[j][k];
            a++;
        }
    }

    // tag=1 for matrix b
    MPI_Recv(&buffer[0][0], block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    for(int j = 0; j < block_size; ++j) {
        for (int k = 0; k < block_size; ++k) {
            *b = buffer[j][k];
            b++;
        }
    }
}

void computePart(const double *a, const double *b, double *c, int block_size) {
    for (int i = 0; i < block_size; ++i) {
        for (int j = 0; j < block_size; ++j) {
            for (int k = 0; k < block_size; ++k) {
                // c[i][j] += a[i][k] * b[k][j]
                *(c + i * block_size + j) += *(a + i * block_size + k) * *(b + k * block_size + j); // TODO: check
            }
        }
    }
}

void sendRowWise(double *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));

    int next_process = rank - width;
    if (next_process < 0) next_process += size - 1;

    MPI_Send(&a[0][0], block_size * block_size, MPI_DOUBLE, next_process, rank, MPI_COMM_WORLD);
}

void receiveRowWise(double *a, int block_size);

void sendColWise(double *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));

    int next_process = rank - 1;

    if (next_process % width == 0) next_process += width;

    MPI_Send(&b[0][0], block_size * block_size, MPI_DOUBLE, next_process, rank, MPI_COMM_WORLD);
}

void receiveColWise(double *b, int block_size);

void submitMatrixPart(double *c, int block_size);

void handleMatrixPart(int blockSize, int width) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double a[blockSize][blockSize];
    double b[blockSize][blockSize];
    double c[blockSize][blockSize];

    receiveMatrixPart(&a[0][0], &b[0][0], blockSize);
    computePart(&a[0][0], &b[0][0], &c[0][0], blockSize);

    for (int i = 0; i < width; ++i) {
        for (int j = 0; j < width; ++j) {
            sendColWise(&b[0][0], blockSize);
            receiveColWise(&b[0][0], blockSize);

            computePart(&a[0][0], &b[0][0], &c[0][0], blockSize);
        }

        sendRowWise(&a[0][0], blockSize);
        receiveRowWise(&a[0][0], blockSize);
    }

    submitMatrixPart(&c[0][0], blockSize);
}

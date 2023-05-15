#include "mpi_mm.h"
#include <mpi.h>
#include <random>
#include <iostream>
#include "mm.h"

void distributeMatrix(const float *a, const float *b, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = int(log2(processes - 1));
    if (width == 0) width = 1;

    int block_size = N / width;
    int cell = width - 1;

    for (int i = 0; i < processes - 1; ++i) {
        // compute cell index for multiplying the matrix block-wise
        if (i % width != 0) {
            cell++;
            if (cell == width) cell = 0;
        }

        // build buffer with square matrix part A
        float buffer[block_size][block_size];

        for(int j = 0; j < block_size; ++j) {
            int row = j + (i / width) * block_size;
            for (int k = 0; k < block_size; ++k) {
                int col = k + cell * block_size;
                buffer[j][k] = *(a + row * N + col);
            }
        }

        // i+1 since main process is 0 ;tag=0 for matrix a
        MPI_Send(&buffer[0][0], block_size * block_size, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD);

        // build buffer with square matrix part B
        for(int j = 0; j < block_size; ++j) {
            int row = j + cell * block_size;
            for (int k = 0; k < block_size; ++k) {
                int col = k + (i % width) * block_size;
                buffer[j][k] = *(b + row * N + col);
            }
        }

        // i+1 since main process is 0 ;tag=1 for matrix b
        MPI_Send(&buffer[0][0], block_size * block_size, MPI_FLOAT, i + 1, 1, MPI_COMM_WORLD);
    }
}

void collectMatrix(float *c, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = int(log2(processes - 1));
    int block_size = width == 0 ? N : N / width;

    auto* buffer = (float*) malloc(block_size * block_size * sizeof(float));

    for (int i = 0; i < processes - 1; ++i) {
        // source must be i + 1 because 0 is the main process
        MPI_Recv(buffer, block_size * block_size, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        int c_ptr = 0;
        if (width > 0) c_ptr = (i / width) * block_size * N + (i % width) * block_size;

        for(int row = 0; row < block_size; ++row) {
            int start_ptr = c_ptr;

            for (int col = 0; col < block_size; ++col) {
                float f = *(buffer + row * block_size + col);
                *(c + c_ptr++) = f;
            }

            c_ptr = start_ptr + N;
        }
    }
}

void receiveMatrixPart(float *a, float *b, int block_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // tag=0 for matrix a
    MPI_Recv(a, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    // tag=1 for matrix b
    MPI_Recv(b, block_size * block_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
}

void computePart(float *a, float *b, float *c, int block_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    multiplyMatrix(a, b, c, block_size);
}

void sendRowWise(float *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));
    int next_process = rank + width;
    if (next_process >= size) next_process -= width * width;

    MPI_Request request;
    MPI_Isend(b, block_size * block_size, MPI_FLOAT, next_process, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void receiveRowWise(float *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));
    int prev_process = rank - width;
    if (prev_process <= 0) prev_process += width * width;

    MPI_Request request;
    MPI_Irecv(b, block_size * block_size, MPI_FLOAT, prev_process, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void sendColWise(float *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));

    int next_process;
    if ((rank - 1) % width == 0) next_process = rank + width - 1;
    else next_process = rank - 1;

    MPI_Request request;
    MPI_Isend(a, block_size * block_size, MPI_FLOAT, next_process, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void receiveColWise(float *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));

    int prev_process;
    if (rank % width == 0) prev_process = rank - width + 1;
    else prev_process = rank + 1;

    MPI_Request request;
    MPI_Irecv(a, block_size * block_size, MPI_FLOAT, prev_process, 0, MPI_COMM_WORLD, &request);
    MPI_Wait(&request, MPI_STATUS_IGNORE);
}

void submitMatrixPart(float *c, int block_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Send(c, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
}

void handleMatrixPart(int N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));
    int block_size = width == 0 ? N : N / width;

    auto* a = (float*) malloc(block_size * block_size * sizeof(float));
    auto* b = (float*) malloc(block_size * block_size * sizeof(float));
    auto* c = (float*) malloc(block_size * block_size * sizeof(float));
    auto* buffer = (float*) malloc(block_size * block_size * sizeof(float));

    // clear c matrix memory
    for (int row = 0; row < block_size; ++row) {
        for (int col = 0; col < block_size; ++col) {
            int c_ptr = row * block_size + col;
            *(c + c_ptr) = 0;
        }
    }

    receiveMatrixPart(a, b, block_size);
    computePart(a, b, c, block_size);

    for (int i = 0; i < width - 1; ++i) {
        if ((rank - 1) % 2 == 0) {
            sendColWise(a, block_size);
            receiveColWise(buffer, block_size);

            auto *temp = a;
            a = buffer;
            buffer = temp;
        } else {
            receiveColWise(buffer, block_size);
            sendColWise(a, block_size);

            auto *temp = a;
            a = buffer;
            buffer = temp;
        }

        if (((rank - 1) / width) % 2 == 0) {
            sendRowWise(b, block_size);
            receiveRowWise(buffer, block_size);

            auto *temp = b;
            b = buffer;
            buffer = temp;
        } else {
            receiveRowWise(buffer, block_size);
            sendRowWise(b, block_size);

            auto *temp = b;
            b = buffer;
            buffer = temp;
        }

        computePart(a, b, c, block_size);
    }

    submitMatrixPart(c, block_size);
}

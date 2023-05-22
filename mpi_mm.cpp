#include "mpi_mm.h"
#include <mpi.h>
#include <random>
#include <iostream>
#include "omp_mm.h"
#include "mm.h"

void distributeMatrix(const float *a, const float *b, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = int(sqrt(processes - 1));
    if (width == 0) width = 1;

    int block_size = N / width;
    int cell = width - 1;

    auto* requests = new MPI_Request[(processes - 1) * 2];
    for (int i = 0; i < processes - 1; ++i) {
        // compute cell index for multiplying the matrix block-wise
        if (i % width != 0) {
            cell++;
            if (cell == width) cell = 0;
        }

        // build buffer with square matrix part A
        float buffer_a[block_size][block_size];

        for(int j = 0; j < block_size; ++j) {
            int row = j + (i / width) * block_size;
            for (int k = 0; k < block_size; ++k) {
                int col = k + cell * block_size;
                buffer_a[j][k] = *(a + row * N + col);
            }
        }

        // i+1 since main process is 0 ;tag=0 for matrix a
        MPI_Isend(&buffer_a[0][0], block_size * block_size, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD, &requests[i * 2]);

        // build buffer with square matrix part B
        float buffer_b[block_size][block_size];

        for(int j = 0; j < block_size; ++j) {
            int row = j + cell * block_size;
            for (int k = 0; k < block_size; ++k) {
                int col = k + (i % width) * block_size;
                buffer_b[j][k] = *(b + row * N + col);
            }
        }

        // i+1 since main process is 0 ;tag=1 for matrix b
        MPI_Isend(&buffer_b[0][0], block_size * block_size, MPI_FLOAT, i + 1, 1, MPI_COMM_WORLD, &requests[i * 2 + 1]);
    }

    for (int i = 0; i < (processes - 1) * 2; i++) {
        MPI_Wait(&requests[i], MPI_STATUS_IGNORE);
    }
}

void collectMatrix(float *c, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = int(sqrt(processes - 1));
    int block_size = width == 0 ? N : N / width;

    float* buffer[processes - 1];
    auto* requests = new MPI_Request[processes - 1];

    for (int i = 0; i < processes - 1; ++i) {
        // source must be i + 1 because 0 is the main process
        buffer[i] = (float*) malloc(block_size * block_size * sizeof(float));

        MPI_Irecv(buffer[i], block_size * block_size, MPI_FLOAT, i + 1, 0, MPI_COMM_WORLD, &requests[i]);
    }

    for (int i = 0; i < processes - 1; ++i) {
        MPI_Wait(&requests[i], MPI_STATUS_IGNORE);

        int c_ptr = 0;
        if (width > 0) c_ptr = (i / width) * block_size * N + (i % width) * block_size;

        for(int row = 0; row < block_size; ++row) {
            int start_ptr = c_ptr;

            for (int col = 0; col < block_size; ++col) {
                float f = *(buffer[i] + row * block_size + col);
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
    MPI_Request req_a;
    MPI_Irecv(a, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, &req_a);

    // tag=1 for matrix b
    MPI_Request req_b;
    MPI_Irecv(b, block_size * block_size, MPI_FLOAT, 0, 1, MPI_COMM_WORLD, &req_b);

    MPI_Wait(&req_a, MPI_STATUS_IGNORE);
    MPI_Wait(&req_b, MPI_STATUS_IGNORE);
}

void computePart(float *a, float *b, float *c, int block_size, bool open_mp) {
    if (open_mp) multiplyMatrixOMP(a, b, c, block_size, 2);
    else multiplyMatrix(a, b, c, block_size);
}

MPI_Request sendRowWise(float *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(sqrt(size - 1));
    int next_process = rank + width;
    if (next_process >= size) next_process -= width * width;

    MPI_Request request;
    MPI_Isend(b, block_size * block_size, MPI_FLOAT, next_process, 0, MPI_COMM_WORLD, &request);
    return request;
}

MPI_Request receiveRowWise(float *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(sqrt(size - 1));
    int prev_process = rank - width;
    if (prev_process <= 0) prev_process += width * width;

    MPI_Request request;
    MPI_Irecv(b, block_size * block_size, MPI_FLOAT, prev_process, 0, MPI_COMM_WORLD, &request);
    return request;
}

MPI_Request sendColWise(float *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(sqrt(size - 1));

    int next_process;
    if ((rank - 1) % width == 0) next_process = rank + width - 1;
    else next_process = rank - 1;

    MPI_Request request;
    MPI_Isend(a, block_size * block_size, MPI_FLOAT, next_process, 0, MPI_COMM_WORLD, &request);
    return request;
}

MPI_Request receiveColWise(float *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(sqrt(size - 1));

    int prev_process;
    if (rank % width == 0) prev_process = rank - width + 1;
    else prev_process = rank + 1;

    MPI_Request request;
    MPI_Irecv(a, block_size * block_size, MPI_FLOAT, prev_process, 0, MPI_COMM_WORLD, &request);
    return request;
}

void submitMatrixPart(float *c, int block_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    MPI_Send(c, block_size * block_size, MPI_FLOAT, 0, 0, MPI_COMM_WORLD);
}

void handleMatrixPart(int N, bool open_mp) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(sqrt(size - 1));
    int block_size = width == 0 ? N : N / width;

    auto* a = (float*) malloc(block_size * block_size * sizeof(float));
    auto* b = (float*) malloc(block_size * block_size * sizeof(float));
    auto* c = (float*) malloc(block_size * block_size * sizeof(float));
    auto* buffer_a = (float*) malloc(block_size * block_size * sizeof(float));
    auto* buffer_b = (float*) malloc(block_size * block_size * sizeof(float));

    // clear c matrix memory
    for (int row = 0; row < block_size; ++row) {
        for (int col = 0; col < block_size; ++col) {
            int c_ptr = row * block_size + col;
            *(c + c_ptr) = 0;
        }
    }

    receiveMatrixPart(a, b, block_size);
    computePart(a, b, c, block_size, open_mp);

    for (int i = 0; i < width - 1; ++i) {
        MPI_Request send_col = sendColWise(a, block_size);
        MPI_Request send_row = sendRowWise(b, block_size);

        MPI_Request recv_col = receiveColWise(buffer_a, block_size);
        MPI_Request recv_row = receiveRowWise(buffer_b, block_size);

        MPI_Wait(&send_col, MPI_STATUS_IGNORE);
        MPI_Wait(&send_row, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_col, MPI_STATUS_IGNORE);
        MPI_Wait(&recv_row, MPI_STATUS_IGNORE);

        auto *temp = a;
        a = buffer_a;
        buffer_a = temp;

        temp = b;
        b = buffer_b;
        buffer_b = temp;

        computePart(a, b, c, block_size, open_mp);
    }

    submitMatrixPart(c, block_size);
}

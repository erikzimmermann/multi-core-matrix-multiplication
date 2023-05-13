#include "mpi_mm.h"
#include <mpi.h>
#include <random>
#include <iostream>

void _printMatrix(double *m, int rows, int cols) {
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << *m << " ";
            m++;
        }
        std::cout << std::endl;
    }
}

// correct
void distributeMatrix(double *a, double *b, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = int(log2(processes - 1));
    if (width == 0) width = 1;

    int block_size = N / width;
    int cell = width - 1;

    _printMatrix(b, N, N);

    for (int i = 0; i < processes - 1; ++i) {
        // determine cell information
        if (i % width != 0) {
            cell++;
            if (cell == width) cell = 0;
        }

        // build buffer with square matrix part A
        double buffer[block_size][block_size];

        for(int j = 0; j < block_size; ++j) {
            int row = j + (i / width) * block_size;
            for (int k = 0; k < block_size; ++k) {
                int col = k + cell * block_size;
                buffer[j][k] = *(a + row * N + col);
            }
        }

        // i+1 since main process is 0 ;tag=0 for matrix a
        MPI_Send(&buffer[0][0], block_size * block_size, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD);
//        std::cout << "a to " << i + 1 << ": " << buffer[0][0] << std::endl;

        // build buffer with square matrix part B
        for(int j = 0; j < block_size; ++j) {
            int row = j + cell * block_size;
            for (int k = 0; k < block_size; ++k) {
                int col = k + (i % width) * block_size;
                buffer[j][k] = *(b + row * N + col);
            }
        }

        // i+1 since main process is 0 ;tag=1 for matrix b
        MPI_Send(&buffer[0][0], block_size * block_size, MPI_DOUBLE, i + 1, 1, MPI_COMM_WORLD);
//        std::cout << "b to " << i + 1 << ": " << buffer[0][0] << std::endl;
    }
}

// correct
void collectMatrix(double *c, int N) {
    int processes;
    MPI_Comm_size(MPI_COMM_WORLD, &processes);

    int width = int(log2(processes - 1));
    int block_size = width == 0 ? N : N / width;

//    std::cout << "collecting" << std::endl;
    for (int i = 0; i < processes - 1; ++i) {
        // build buffer with square matrix part C
        double buffer[block_size][block_size];

        // source must be i + 1 because 0 is the main process
        MPI_Recv(&buffer[0][0], block_size * block_size, MPI_DOUBLE, i + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//        std::cout << "from " << i + 1 << ": " << buffer[0][0] << std::endl;

        int c_ptr = 0;
        if (width > 0) c_ptr = (i / width) * block_size * N + (i % width) * block_size;

        for(int row = 0; row < block_size; ++row) {
            int start_ptr = c_ptr;

            for (int col = 0; col < block_size; ++col) {
                *(c + c_ptr++) = buffer[row][col];
            }

            c_ptr = start_ptr + N;
        }
    }
}

// correct
void receiveMatrixPart(double *a, double *b, int block_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // tag=0 for matrix a
    MPI_Recv(a, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    std::cout << "a in " << rank << ": " << *a << std::endl;

    // tag=1 for matrix b
    MPI_Recv(b, block_size * block_size, MPI_DOUBLE, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
//    std::cout << "b in " << rank << ": " << *b << std::endl;

}

// correct
void computePart(double *a, double *b, double *c, int block_size) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 1) {
        std::cout << "Start computation" << std::endl;
    }

    for (int row = 0; row < block_size; ++row) {
        for (int col = 0; col < block_size; ++col) {
            int a_ptr = row * block_size;
            int b_ptr = col;
            int c_ptr = row * block_size + col;

            for (int i = 0; i < block_size; ++i) {
                *(c + c_ptr) += *(a + a_ptr++) * *(b + b_ptr);
                b_ptr += block_size;
            }
        }
    }

    if (rank == 1) {
        std::cout << "Done with comp." << std::endl;
    }
}

void sendRowWise(double *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));
    int next_process = rank + width;
    if (next_process >= size) next_process -= width * width;

//    std::cout << "sendRowWise " << rank << " -> " << next_process << std::endl;
    MPI_Request request;
    MPI_Isend(b, block_size * block_size, MPI_DOUBLE, next_process, 0, MPI_COMM_WORLD, &request);
}

void receiveRowWise(double *b, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));
    int prev_process = rank - width;
    if (prev_process <= 0) prev_process += width * width;

//    std::cout << "receiveRowWise " << rank << " <- " << prev_process << " (" << size << ")" << std::endl;
    MPI_Recv(b, block_size * block_size, MPI_DOUBLE, prev_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank == 3) {
        std::cout << "get row for b from " << prev_process << ": " << *b << std::endl;
        _printMatrix(b, block_size, block_size);
    }
}

void sendColWise(double *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));

    int next_process;
    if ((rank - 1) % width == 0) next_process = rank + width - 1;
    else next_process = rank - 1;

//    std::cout << "SendColWise " << rank << " -> " << next_process << std::endl;

    MPI_Request request;
    MPI_Isend(a, block_size * block_size, MPI_DOUBLE, next_process, 0, MPI_COMM_WORLD, &request);
}

void receiveColWise(double *a, int block_size) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));

    int prev_process;
    if (rank % width == 0) prev_process = rank - width + 1;
    else prev_process = rank + 1;

//    std::cout << "receiveColWise " << rank << " <- " << prev_process << std::endl;
    MPI_Recv(a, block_size * block_size, MPI_DOUBLE, prev_process, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    if (rank == 3) {
        std::cout << "get col for a from " << prev_process << ": " << *a << std::endl;
        _printMatrix(a, block_size, block_size);
    }
}

void submitMatrixPart(double *c, int block_size) {
    MPI_Send(c, block_size * block_size, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
}

void handleMatrixPart(int N) {
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int width = int(log2(size - 1));
    int block_size = width == 0 ? N : N / width;
    if (rank == 1) {
        std::cout << "width=" << width << " block_size=" << block_size << std::endl;
    }

    double a[block_size][block_size];
    double b[block_size][block_size];
    double c[block_size][block_size];

    receiveMatrixPart(&a[0][0], &b[0][0], block_size);
    computePart(&a[0][0], &b[0][0], &c[0][0], block_size);

    if (rank == 3) {
        std::cout << "get col for a from " << 0 << ": " << *a << std::endl;
        _printMatrix(&a[0][0], block_size, block_size);
        std::cout << "get row for b from " << 0 << ": " << *a << std::endl;
        _printMatrix(&b[0][0], block_size, block_size);
    }

    for (int i = 0; i < width - 1; ++i) {
        if (rank == 1) {
            std::cout << "it: " << i << std::endl;
        }

        sendColWise(&a[0][0], block_size);
        if (rank == 1) {
            std::cout << "sendColWise" << std::endl;
        }
        sendRowWise(&b[0][0], block_size);
        if (rank == 1) {
            std::cout << "sendRowWise" << std::endl;
        }

        receiveColWise(&a[0][0], block_size);
        if (rank == 1) {
            std::cout << "receiveColWise" << std::endl;
        }
        receiveRowWise(&b[0][0], block_size);
        if (rank == 1) {
            std::cout << "receiveRowWise" << std::endl;
        }

        if (rank == 1) {
            std::cout << "it: " << i << " SENT" << std::endl;
        }

        computePart(&a[0][0], &b[0][0], &c[0][0], block_size);
    }

    submitMatrixPart(&c[0][0], block_size);
}

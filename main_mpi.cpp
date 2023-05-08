#include <iostream>
#include <mpi.h>

const int N = 1000; // size of matrices

int main(int argc, char** argv) {
    int num_procs, rank;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // initialize matrices A and B
    int A[N][N];
    int B[N][N];
    int C[N][N] = {0}; // initialize result matrix to zero

    // fill matrices A and B with random values
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            A[i][j] = rand() % 100;
            B[i][j] = rand() % 100;
        }
    }

    // start timer
    double start_time = MPI_Wtime();

    // perform matrix multiplication using MPI
    for (int i = rank; i < N; i += num_procs) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // sum the results from all processes
    MPI_Allreduce(MPI_IN_PLACE, C, N * N, MPI_INT, MPI_SUM, MPI_COMM_WORLD);

    // stop timer
    double end_time = MPI_Wtime();

    // print result matrix
    if (rank == 0) {
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                std::cout << C[i][j] << " ";
            }
            std::cout << std::endl;
        }
    }

    // print elapsed time
    if (rank == 0) {
        std::cout << "Elapsed time: " << end_time - start_time << " seconds" << std::endl;
    }

    MPI_Finalize();

    return 0;
}

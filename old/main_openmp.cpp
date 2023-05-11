#include <iostream>
#include <omp.h>

const int N = 410; // size of matrices

int main() {
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
    double start_time = omp_get_wtime();

    // perform matrix multiplication in parallel using OpenMP
#pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                C[i][j] += A[i][k] * B[k][j];
            }
        }
    }

    // stop timer
    double end_time = omp_get_wtime();

    // print result matrix
//    for (int i = 0; i < N; ++i) {
//        for (int j = 0; j < N; ++j) {
//            std::cout << C[i][j] << " ";
//        }
//        std::cout << std::endl;
//    }

    // print elapsed time
    std::cout << "Elapsed time: " << end_time - start_time << " seconds" << std::endl;

    return 0;
}

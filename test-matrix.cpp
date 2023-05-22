#include <iostream>
#include <chrono>
#include <mpi.h>
#include "mm.h"
#include "mpi_mm.h"
#include "omp_mm.h"
#include <random>

std::seed_seq SEED{42};

void fillRandomly(float *a, float *b, float *c, int n) {
    std::default_random_engine generator(SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < n * n; i++) {
        *(a + i) = distribution(generator);
        *(b + i) = distribution(generator);
        *(c + i) = 0;  // clear matrix c
    }
}

void printMatrix(float *m, int rows, int cols) {
    int start_row = 0;
    int start_col = 0;
    int size = 16;

    double sum = 0;
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (i >= start_row && i < start_row + size && j >= start_col && j < start_col + size) {
                std::cout << *m << " ";
                sum += *m;
            }
            m++;
        }
        if (i >= start_row && i < start_row + size) std::cout << std::endl;
    }
    std::cout << "Sum=" << sum << std::endl;
}

void mm_naive(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            float *cptr = c + i * n + j;
            float *aptr = a + i * n;
            float *bptr = b + j;

            for (int k = 0; k < n; ++k) {
                *cptr += *(aptr++) * *bptr;
                bptr += n;
            }
        }
    }
}

void compute_openmp(float *a, float *b, float *c, int n, int threads) {
    multiplyMatrixOMP(a, b, c, n, threads, 1);
}

double calculateChecksum(const float *c, int n) {
    double sum = 0;

    for (int i = 0; i < n * n; ++i) {
        sum += *(c + i);
    }

    return sum;
}

int compute_mpi(int argc, char* argv[], float *a, float *b, float *c, int n, bool open_mp) {
    int rank, size;

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    int processes = size - 1;
    if (sqrt(processes) - double(int(sqrt(processes))) > 0.1 || size < 5) {
        if (rank == 0) std::cerr << "The number of processes (" << processes << ", " << size << ") must be the result of x*x+1 and greater equals 5." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (n % int(sqrt(processes)) != 0) {
        if (rank == 0) std::cerr << "n and the amount of processes must be compliant so that result of n / sqrt(processes - 1) is an integer." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (rank == 0) {
        fillRandomly(a, b, c, n);

        auto start_time = std::chrono::system_clock::now();
        distributeMatrix(a, b, n);
        collectMatrix(c, n);

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        auto sum = short(calculateChecksum(c, n));
        std::cout << elapsed_seconds.count() << "; checksum: " << sum << std::endl;

//        printMatrix(c, n, n);
    } else {
        handleMatrixPart(n, open_mp);
    }

    MPI_Finalize();
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Parameters: <n>, <seq, naive, omp, mpi>" << std::endl;
        return 1;
    }

    int n = std::stoi(argv[1]);
    std::string type = std::string(argv[2]);

    auto* a = (float*) malloc(n * n * sizeof(float));
    auto* b = (float*) malloc(n * n * sizeof(float));
    auto* c = (float*) malloc(n * n * sizeof(float));

    if (type == "mpi") {
        return compute_mpi(argc, argv, a, b, c, n, false);
    } else if (type == "mpiomp") {
        return compute_mpi(argc, argv, a, b, c, n, true);
    } else {
        fillRandomly(a, b, c, n);
        auto start_time = std::chrono::system_clock::now();

        if (type == "seq") {
            multiplyMatrix(a, b, c, n);
        } else if (type == "naive") {
            mm_naive(a, b, c, n);
        } else if (type.compare(0, 3, "omp") == 0) {
            if (type.length() > 3) {
                size_t dashPos = type.find('-');
                if (dashPos != std::string::npos) {
                    std::string numStr = type.substr(dashPos + 1);
                    int threads = std::stoi(numStr);
                    compute_openmp(a, b, c, n, threads);
                } else {
                    std::cerr << "Specify type: omp or omp-<threads>" << std::endl;
                    return 1;
                }
            } else compute_openmp(a, b, c, n, 0);
        } else {
            std::cerr << "Specify type: seq, naive, omp, omp-4, omp-8, mpi" << std::endl;
            return 1;
        }

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        auto sum = short(calculateChecksum(c, n));
        std::cout << elapsed_seconds.count() << "; checksum: " << sum << std::endl;

//        printMatrix(c, n, n);
    }

    return 0;
}

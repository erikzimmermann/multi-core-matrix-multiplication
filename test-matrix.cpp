#include <iostream>
#include <chrono>
#include <mpi.h>
#include <random>
#include "mpi_mm.h"
#include "mm.h"

std::seed_seq SEED{42};

void fillRandomly(float *a, float *b, float *c, int n) {
    std::default_random_engine generator(SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            *(a + i * n + j) = distribution(generator);
            *(b + i * n + j) = distribution(generator);
            *(c + i * n + j) = 0;
        }
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

void mm_openmp(float *a, float *b, float *c, int n, int a_row, int b_col, int block_size) {
    int remaining = std::min(n - a_row, block_size);

    for (int i = 0; i < remaining; i++) {
        for (int j = 0; j < remaining; j++) {
            float *cPtr = c + (a_row + i) * n + b_col + j;
            float *aPtr = a + (a_row + i) * n;
            float *bPtr = b + b_col + j;

            for (int k = 0; k < n; ++k) {
                *cPtr += *(aPtr++) * *bPtr;
                bPtr += n;
            }
        }
    }
}

void compute_openmp(float *a, float *b, float *c, int n) {
    int block_size = 32;
    if (n % block_size != 0) {
        block_size = 25;
        if (n % block_size != 0) {
            std::cerr << "Invalid block size. n=" << n << ", block_size=" << block_size << std::endl;
            return;
        }
    }

    int iter = std::ceil(float(n) / float(block_size));

    #pragma omp parallel default(none) shared(a, b, c, n, iter, block_size)
    #pragma omp single
    {
        for (int i = 0; i < iter; ++i) {
            for (int j = 0; j < iter; ++j) {
                #pragma omp task
                mm_openmp(a, b, c, n, i * block_size, j * block_size, block_size);
            }
        }
    }
}

double calculateChecksum(const float *c, int n) {
    double sum = 0;

    for (int i = 0; i < n * n; ++i) {
        sum += *(c + i);
    }
    
    return sum;
}

int compute_mpi(int argc, char* argv[], float *a, float *b, float *c, int n) {
    int rank, size;

    MPI_Init(&argc, &argv); // Initialize MPI environment
    MPI_Comm_rank(MPI_COMM_WORLD, &rank); // Get the rank of the current process
    MPI_Comm_size(MPI_COMM_WORLD, &size); // Get the total number of processes

    int processes = size - 1;
    if (sqrt(processes) != double(int(sqrt(processes))) || processes < 5) {
        if (rank == 0) std::cerr << "The number of processes must be the result of x*x+1 and greater equals 5." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }

    if (n % int(sqrt(processes)) != 0|| int(sqrt(processes)) % 2 != 0) {
        if (rank == 0) std::cerr << "n and the amount of processes must be compliant so that result of n / sqrt(processes - 1) is an integer." << std::endl;
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    
    if (rank == 0) {
        fillRandomly(a, b, c, n);

        // distribute matrix by using the pointers to the matrices a and b
        distributeMatrix(a, b, n);

        auto start_time = std::chrono::system_clock::now();
        collectMatrix(c, n);

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        auto sum = short(calculateChecksum(c, n));
        std::cout << elapsed_seconds.count() << "; checksum: " << sum << std::endl;
    } else {
        handleMatrixPart(n);
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
        return compute_mpi(argc, argv, a, b, c, n);
    } else {
        fillRandomly(a, b, c, n);
        auto start_time = std::chrono::system_clock::now();

        if (type == "seq") {
            multiplyMatrix(a, b, c, n);
        } else if (type == "naive") {
            mm_naive(a, b, c, n);
        } else if (type == "omp") {
            compute_openmp(a, b, c, n);
        } else {
            std::cerr << "Specify type: seq, naive, omp, mpi" << std::endl;
            return 1;
        }

        auto end_time = std::chrono::system_clock::now();
        std::chrono::duration<double> elapsed_seconds = end_time - start_time;

        auto sum = short(calculateChecksum(c, n));
        std::cout << elapsed_seconds.count() << "; checksum: " << sum << std::endl;
    }

    return 0;
}

#include <iostream>
#include <chrono>
#include <random>

const double CHECK_SUM = -12395;
std::seed_seq SEED{42};

const int N = 2048;
const int THRESHOLD = 32768;
const int BLOCK_SIZE = 32;

double a[N][N];
double b[N][N];
double c[N][N];

void fillRandomly() {
    std::default_random_engine generator(SEED);
    std::uniform_real_distribution<float> distribution(0.0f, 1.0f);

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            a[i][j] = distribution(generator);
            b[i][j] = distribution(generator);
        }
    }
}

void mm_naive() {
    int check = N / 100;

    for (int i = 0; i < N; ++i) {
        if (i % check == 0) {
            std::cout << "it " << i << std::endl;
        }

        for (int j = 0; j < N; ++j) {
            for (int k = 0; k < N; ++k) {
                // compute c matrix from a and b
                c[i][j] += a[i][k] * b[k][j];
            }
        }
    }
}

void mm_sequential(int crow, int ccol,
                   int arow, int acol,
                   int brow, int bcol,
                   int l, int m, int n) {
    int lhalf[3], mhalf[3], nhalf[3];
    int i, j, k;
    double *aptr, *bptr, *cptr;

    if (m * n > THRESHOLD) {
        lhalf[0] = 0; lhalf[1] = l/2; lhalf[2] = l - l/2;
        mhalf[0] = 0; mhalf[1] = m/2; mhalf[2] = l - m/2;
        nhalf[0] = 0; nhalf[1] = n/2; nhalf[2] = l - n/2;

        for (i = 0; i < 2; i++) {
            for (j = 0; j < 2; j++) {
                for (k = 0; k < 2; k++) {
                    mm_sequential(crow + lhalf[i], ccol + mhalf[j],
                                  arow + lhalf[i], acol + mhalf[k],
                                  brow + mhalf[k], bcol + nhalf[j],
                                  lhalf[i + 1], mhalf[k + 1], nhalf[k + 1]);
                }
            }
        }
    } else {
        for (i = 0; i < l; i++) {
            for (j = 0; j < n; j++) {
                cptr = &c[crow + i][ccol + j];
                aptr = &a[arow + i][acol];
                bptr = &b[brow][bcol + j];

                for (k = 0; k < m; k++) {
                    *cptr += *(aptr++) * *bptr;
                    bptr += N;
                }
            }
        }
    }
}

void compute_sequential() {
    mm_sequential(0, 0, 0, 0, 0, 0, N, N, N);
}

void mm_openmp(int aRow, int bCol) {
    int remaining = std::min(N - aRow, BLOCK_SIZE);

    for (int i = 0; i < remaining; i++) {
        for (int j = 0; j < remaining; j++) {
            double *cPtr = &c[aRow + i][bCol + j];
            double *aPtr = &a[aRow + i][0];
            double *bPtr = &b[0][bCol + j];

            for (int k = 0; k < N; ++k) {
                *cPtr += *(aPtr++) * *bPtr;
                bPtr += N;
            }
        }
    }
}

void compute_openmp() {
    int iter = std::ceil(float(N) / float(BLOCK_SIZE));
    std::cout << iter * iter << " jobs" << std::endl;

    #pragma omp parallel default(none) shared(iter)
    #pragma omp single
    {
        for (int i = 0; i < iter; ++i) {
            for (int j = 0; j < iter; ++j) {
                #pragma omp task
                mm_openmp(i * BLOCK_SIZE, j * BLOCK_SIZE);
            }
        }
    }
}

short calculateChecksum() {
    short sum = 0;

    for (auto &i: c) {
        for (auto &j: i) {
            sum += j * 1000;  // ignore casting issues
        }
    }

    return sum;
}

void printMatrix(double m[N][N]) {
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << m[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char* argv[]) {
    fillRandomly();

    auto start_time = std::chrono::system_clock::now();

    /*
     * seq
     *
     * 1024:32768
     * sum: -32072 (Correct)
     * Elapsed time: 5.10194 seconds
     *
     * 2048:32768
     * sum: -12395 (Correct)
     * Elapsed time: 40.8208 seconds
     *
     * #####
     *
     * naive
     *
     * 1024
     * sum: -32072 (Correct)
     * Elapsed time: 9.63945 seconds
     *
     * 2048
     * sum: -12395 (Correct)
     * Elapsed time: 280.025 seconds
     *
     * #####
     *
     * openmp
     *
     * 1024:64
     * sum: -32072 (Correct)
     * Elapsed time: 0.858777 seconds
     *
     * 2048:64
     * sum: -12395 (Correct)
     * Elapsed time: 27.167 seconds
     *
     * 2048:32
     * sum: -12395 (Correct)
     * Elapsed time: 27.1543 seconds
     *
     * 2048:128
     * sum: -12395 (Correct)
     * Elapsed time: 29.5544 seconds
     */

    // check if argv[0] is "seq"
    if (argc > 1 && std::string(argv[1]) == "seq") {
        compute_sequential();
    } else if (argc > 1 && std::string(argv[1]) == "naive") {
        mm_naive();
    } else if (argc > 1 && std::string(argv[1]) == "omp") {
        compute_openmp();
    } else {
        std::cout << "Specify type: seq, naive, omp" << std::endl;
        return 1;
    }

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    short sum = calculateChecksum();
    std::cout << "sum: " << sum << (sum == CHECK_SUM ? " (Correct)" : " (INCORRECT)") << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}

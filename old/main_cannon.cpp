#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>

// required time: 83.0566 seconds
const double CHECK_SUM = -13325;
std::seed_seq SEED{42};

const int N = 3000;
const int BLOCK_SIZE = 500;

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

void mm(int aRow, int bCol) {
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

    int tid = omp_get_thread_num();
    std::cout << "Done " << tid << std::endl;
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

int main() {
    fillRandomly();

    auto start_time = std::chrono::system_clock::now();


    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    short sum = calculateChecksum();
    std::cout << "sum: " << sum << (sum == CHECK_SUM ? " (Correct)" : " (INCORRECT)") << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

//        printMatrix(c);

//    2.30518 2.07573 3.02842 2.14393 2.55232 1.76918 2.25912 3.16877 2.32598 1.86034
//    2.63489 2.46564 4.01446 2.27187 2.83082 2.54348 2.07249 2.81753 2.85659 2.51156
//    2.19033 2.38243 2.78196 1.96973 2.9126 1.70567 2.15178 3.32892 2.41793 2.16854
//    1.93505 2.23455 3.60658 2.35129 2.44122 2.00202 2.12657 2.85117 2.41426 2.12343
//    1.98447 2.45031 4.14643 2.86672 2.99451 2.10801 2.1705 3.54782 2.99006 2.62987
//    3.27872 3.42899 3.99312 3.00246 3.41575 2.61585 2.80487 3.83812 3.63848 2.71688
//    2.04964 2.22392 2.32521 1.66711 2.08852 1.38859 1.50813 2.11736 2.15142 1.5662
//    2.71104 2.75032 4.35981 2.70453 3.14078 2.32951 2.52032 3.42665 3.07184 2.62292
//    1.74068 1.74449 1.82961 1.23409 1.6509 1.27002 1.20558 2.23735 2.05073 1.12399
//    2.61973 2.46081 3.03519 2.082 2.78924 2.0134 2.09583 2.72937 2.76786 2.33105

    return 0;
}

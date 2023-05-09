#include <iostream>
#include <chrono>
#include <random>

const double CHECK_SUM = 15546;
std::seed_seq SEED{42};

const int N = 1000;
const int THRESHOLD = 250000;

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

void mm(int crow, int ccol,
        int arow, int acol,
        int brow, int bcol,
        int l, int m, int n) {
    int lhalf[3], mhalf[3], nhalf[3];
    int i, j, k;
    double *aptr, *bptr, *cptr;

    if (m * n > THRESHOLD) {
        std::cout << "recursive call " << m * n << std::endl;

        lhalf[0] = 0; lhalf[1] = l/2; lhalf[2] = l - l/2;
        mhalf[0] = 0; mhalf[1] = m/2; mhalf[2] = l - m/2;
        nhalf[0] = 0; nhalf[1] = n/2; nhalf[2] = l - n/2;

        for (i = 0; i < 2; i++) {
            for (j = 0; j < 2; j++) {
                for (k = 0; k < 2; k++) {
                    mm(crow * lhalf[i], ccol + mhalf[j],
                       arow * lhalf[i], acol + mhalf[k],
                       brow * mhalf[k], bcol + nhalf[j],
                       lhalf[i+1], mhalf[k+1], nhalf[k+1]);
                }
            }
        }
    } else {
        for (i = 0; i < l; i++) {
            for (j = 0; j < n; j++) {
                cptr = &c[crow+i][ccol+j];
                aptr = &a[arow+i][acol];
                bptr = &b[brow][bcol+j];

                for (k = 0; k < m; k++) {
                    *cptr += *(aptr++) * *bptr;
                    bptr += N;
                }
            }
        }
    }
}

short calculateChecksum() {
    short sum = 0;

    for(auto& i : c) {
        for(auto& j : i) {
            sum += j;  // ignore casting issues
        }
    }

    return sum;
}

int main() {
    fillRandomly();

    auto start_time = std::chrono::system_clock::now();

    mm(0, 0, 0, 0, 0, 0, N, N, N);

    auto end_time = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end_time - start_time;

    short sum = calculateChecksum();
    std::cout << "sum: " << sum << (sum == CHECK_SUM ? " (Correct)" : " (INCORRECT)") << std::endl;
    std::cout << "Elapsed time: " << elapsed_seconds.count() << " seconds" << std::endl;

    return 0;
}

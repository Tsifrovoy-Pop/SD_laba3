extern "C" {
#include <cblas.h>
}
#include <iostream>
#include <complex>
#include <vector>
#include <chrono>
#include <random>
#include <cmath>
#include <algorithm>

using namespace std;
using namespace chrono;

using Complex = complex<double>;

void multiply_naive_ijk(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C, int n) {
    fill(C.begin(), C.end(), Complex(0, 0));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Complex sum = 0;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

void multiply_optimized_ikj(const vector<Complex>& A, const vector<Complex>& B, vector<Complex>& C, int n) {
    fill(C.begin(), C.end(), Complex(0, 0));
    for (int i = 0; i < n; i++) {
        for (int k = 0; k < n; k++) {
            Complex aik = A[i * n + k];
            const Complex* b_row = &B[k * n];
            Complex* c_row = &C[i * n];
            for (int j = 0; j < n; j++) {
                c_row[j] += aik * b_row[j];
            }
        }
    }
}

void multiply_blocked_omp_optimized(const Complex* __restrict A,
    const Complex* __restrict B,
    Complex* __restrict C,
    int n)
{
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n * n; ++i) {
        C[i] = Complex(0.0, 0.0);
    }

    const int BS = 64;

#pragma omp parallel for collapse(2) schedule(dynamic, 1)
    for (int ii = 0; ii < n; ii += BS) {
        for (int jj = 0; jj < n; jj += BS) {
            for (int kk = 0; kk < n; kk += BS) {

                int i_end = min(ii + BS, n);
                int j_end = min(jj + BS, n);
                int k_end = min(kk + BS, n);

                for (int i = ii; i < i_end; ++i) {
                    Complex* __restrict c_row = &C[i * n];

                    for (int k = kk; k < k_end; ++k) {
                        const Complex aik = A[i * n + k];
                        const Complex* __restrict b_row = &B[k * n];

                        int j = jj;

                        for (; j <= j_end - 2; j += 2) {
                            Complex c0 = c_row[j];
                            Complex c1 = c_row[j + 1];

                            c0 += aik * b_row[j];
                            c1 += aik * b_row[j + 1];

                            c_row[j] = c0;
                            c_row[j + 1] = c1;
                        }

                        for (; j < j_end; ++j) {
                            c_row[j] += aik * b_row[j];
                        }
                    }
                }

            }
        }
    }
}

double matrix_diff_norm(const vector<Complex>& A, const vector<Complex>& B) {
    double sum = 0;
    for (size_t i = 0; i < A.size(); i++) {
        sum += norm(A[i] - B[i]);
    }
    return sqrt(sum);
}

int main() {
    system("chcp 65001 > nul");
    setlocale(LC_ALL, "Russian");

    cout << "=== Проверка конфигурации сборки ===\n";
#ifdef _OPENMP
    cout << "OpenMP успешно активирован в проекте.\n\n";
#else
    cout << "? ВНИМАНИЕ: OpenMP выключен. Включите его в настройках проекта Visual Studio!\n\n";
#endif

    _putenv("OPENBLAS_NUM_THREADS=1");
    _putenv("GOTO_NUM_THREADS=1");
    _putenv("OMP_NUM_THREADS=1");

    const int n = 4096;
    const long long ops = 2LL * n * n * n;

    cout << "Размер матрицы: " << n << "x" << n << "\n";
    cout << "Сложность алгоритма (c): " << ops << " операций\n\n";

    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> dis(-1.0, 1.0);

    auto gen_mat = [&]() {
        vector<Complex> mat(n * n);
        for (auto& val : mat) val = Complex(dis(gen), dis(gen));
        return mat;
        };

    cout << "Генерация случайных матриц... ";
    auto A = gen_mat();
    auto B = gen_mat();
    vector<Complex> C1(n * n), C2(n * n), C3(n * n), C4(n * n);
    cout << "готово.\n\n";

    cout << "Запуск Варианта 1 (Классический i-j-k)... ";
    auto start = high_resolution_clock::now();
    multiply_naive_ijk(A, B, C1, n);
    auto end = high_resolution_clock::now();
    double t1 = duration<double>(end - start).count();
    double mflops1 = (ops / t1) * 1e-6;
    cout << "готово.\n";
    cout << "Время: " << t1 << " сек | " << mflops1 << " MFLOPS\n\n";

    cout << "Запуск Варианта 2 (cblas_zgemm из библиотеки BLAS)... ";
    Complex alpha = 1.0;
    Complex beta = 0.0;

    start = high_resolution_clock::now();
    cblas_zgemm(
        CblasRowMajor, CblasNoTrans, CblasNoTrans,
        n, n, n,
        &alpha,
        A.data(), n,
        B.data(), n,
        &beta,
        C4.data(), n
    );
    end = high_resolution_clock::now();
    double t4 = duration<double>(end - start).count();
    double mflops4 = (ops / t4) * 1e-6;
    cout << "готово.\n";
    cout << "Время: " << t4 << " сек | " << mflops4 << " MFLOPS\n\n";

    cout << "Запуск последовательного алгоритма i-k-j... ";
    start = high_resolution_clock::now();
    multiply_optimized_ikj(A, B, C2, n);
    end = high_resolution_clock::now();
    double t2 = duration<double>(end - start).count();
    double mflops2 = (ops / t2) * 1e-6;
    cout << "готово.\n";
    cout << "Время: " << t2 << " сек | " << mflops2 << " MFLOPS\n";
    cout << "Проверка алгоритма i-k-j: " << (matrix_diff_norm(C2, C4) < 1e-9 ? "Успешно" : "Ошибка") << "\n\n";

    cout << "Запуск Варианта 3 (Блочный + OpenMP)... ";
    start = high_resolution_clock::now();
    multiply_blocked_omp_optimized(A.data(), B.data(), C3.data(), n);
    end = high_resolution_clock::now();
    double t3 = duration<double>(end - start).count();
    double mflops3 = (ops / t3) * 1e-6;
    cout << "готово.\n";
    cout << "Время: " << t3 << " сек | " << mflops3 << " MFLOPS\n";
    cout << "Проверка Варианта 3: " << (matrix_diff_norm(C3, C4) < 1e-9 ? "Успешно" : "Ошибка") << "\n\n";

    cout << "=== ИТОГОВОЕ СРАВНЕНИЕ ===\n";
    double perf_ratio = (mflops3 / mflops4) * 100.0;
    cout << "1. Наивный (i-j-k):      " << mflops1 << " MFLOPS (База, 1.00x)\n";
    cout << "2. Порядок i-k-j:        " << mflops2 << " MFLOPS (" << (mflops2 / mflops1) << "x ускорение)\n";
    cout << "3. Блочный + OpenMP:     " << mflops3 << " MFLOPS (" << (mflops3 / mflops1) << "x ускорение)\n";
    cout << "Эффективность Варианта 3 относительно Варианта 2 (BLAS): " << perf_ratio << " %\n\n";

    if (perf_ratio >= 30.0) {
        cout << "Критерий выполнен: скорость алгоритма выше требуемых 30% от BLAS.\n";
    }
    else {
        cout << "Критерий не выполнен: скорость алгоритма ниже требуемых 30% от BLAS.\n";
    }

    return 0;
}

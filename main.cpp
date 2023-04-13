#include <iostream>
#include "Matrix.h"
#include <chrono>
#include <mpi.h>
#include "matrix_mul_pthreads.h"
#include "matrix_mul_mpi.h"
#include "utils.h"
#include "test.h"

using namespace std;
using namespace std::chrono;

int main(int argc, char *argv[]) {
    int lower_bounds = 1000;
    int upper_bounds = 1000;
    int warmup = 10;
    int threads_count = atoi(argv[1]);
    int rows = atoi(argv[2]);
    int cols = atoi(argv[2]);
    int test = atoi(argv[3]);
    char* save_path = argv[4];
    long* times_seq = new long[test];
    long* times_pthread = new long[test];
    long* times_mpi = new long[test];
    Matrix matrix_seq, matrix_pthread, matrix_mpi;
    std::chrono::high_resolution_clock::time_point start, stop, start_pthread, stop_pthread, start_mpi, stop_mpi;
    std::chrono::system_clock::duration duration, duration_pthread, duration_mpi;

    int rank, num_processes;
    Matrix matrix1, matrix2;

    MPI_Init(nullptr, nullptr);
    MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == 0) {
        printf("==============================================================\n");
        printf("Running Tests\n");
        test_square_matrix();
        test_random_size_matrix();
        printf("The tests have been completed\n");
        printf("==============================================================\n");
    }

    double** mat1;
    double** mat2;
    allocateSpace(&mat1, rows, cols);
    allocateSpace(&mat2, rows, cols);

    if (rank == 0) {
        generate_int_matrix_array(mat1, rows, cols, lower_bounds, upper_bounds);
        generate_int_matrix_array(mat2, rows, cols, lower_bounds, upper_bounds);

        matrix1 = Matrix(mat1, rows, cols);
        matrix2 = Matrix(mat2, rows, cols);

        printf("Compare time\n");
        start = high_resolution_clock::now();
        matrix_seq = matrix1 * matrix2;
        stop = high_resolution_clock::now();

        duration = duration_cast<microseconds>(stop.time_since_epoch() - start.time_since_epoch());
        cout << "Time taken for one run sequential: " << duration.count() << " microseconds" << endl;

        start_pthread = high_resolution_clock::now();
        matrix_pthread = multiply_pthread(matrix1, matrix2, threads_count);
        stop_pthread = high_resolution_clock::now();

        duration_pthread = duration_cast<microseconds>(stop_pthread - start_pthread);
        cout << "Time taken for one run pthread: " << duration_pthread.count() << " microseconds" << endl;

        double performance_boost_pthread = (double) duration.count() / (double) duration_pthread.count();
        cout << "Performance boost for pthread for one run: " << performance_boost_pthread << endl;
        printf("==============================================================\n");
    }

    MPI_Bcast(&(mat1[0][0]), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, rows, cols);
    matrix2 = Matrix(mat2, rows, cols);

    double** result_mpi;
    allocateSpace(&result_mpi, rows, cols);

    if (rank == 0) {
        start_mpi = high_resolution_clock::now();
    }

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_mpi = Matrix(result_mpi, rows, cols);
        stop_mpi = high_resolution_clock::now();

        duration_mpi = duration_cast<microseconds>(stop_mpi - start_mpi);
        cout << "Time taken for one run mpi: " << duration_mpi.count() << " microseconds" << endl;

        double performance_boost_mpi = (double) duration.count() / (double) duration_mpi.count();
        cout << "Performance boost for mpi for one run: " << performance_boost_mpi << endl;
        printf("==============================================================\n");

        bool is_equal_pthread = (matrix_seq == matrix_pthread);
        cout << "Sequential and pthread matrices are equals: " << std::boolalpha << is_equal_pthread << endl;

        bool is_equal_mpi = (matrix_seq == matrix_mpi);
        cout << "Sequential and mpi matrices are equals: " << std::boolalpha << is_equal_mpi << endl;
        printf("==============================================================\n");
    }

    if (rank == 0) {
        for (int i = 0; i < test; i++) {
            auto start_test = high_resolution_clock::now();
            matrix_seq = matrix1 * matrix2;
            auto stop_test = high_resolution_clock::now();

            auto duration_test = duration_cast<microseconds>(stop_test - start_test);

            long duration_long = duration_test.count();
            times_seq[i] = duration_long;
        }

        for (int i = 0; i < test; i++) {
            auto start_pthread_test = high_resolution_clock::now();
            matrix_pthread = multiply_pthread(matrix1, matrix2, threads_count);
            auto stop_pthread_test = high_resolution_clock::now();

            auto duration_pthread_test = duration_cast<microseconds>(stop_pthread_test - start_pthread_test);

            long duration_pthread_long = duration_pthread_test.count();
            times_pthread[i] = duration_pthread_long;
        }
    }

    for (int i = 0; i < test; i++) {
        MPI_Bcast(&(mat1[0][0]), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Bcast(&(mat2[0][0]), rows * cols, MPI_DOUBLE, 0, MPI_COMM_WORLD);

        matrix1 = Matrix(mat1, rows, cols);
        matrix2 = Matrix(mat2, rows, cols);

        double** result_mpi_test;
        allocateSpace(&result_mpi_test, rows, cols);

        if (rank == 0) {
            start_mpi = high_resolution_clock::now();
        }

        multiply_mpi(matrix1, matrix2, result_mpi_test, rank, num_processes);

        if (rank == 0) {
            matrix_mpi = Matrix(result_mpi_test, rows, cols);
            stop_mpi = high_resolution_clock::now();

            auto duration_mpi_test = duration_cast<microseconds>(stop_mpi - start_mpi);

            long duration_mpi_long = duration_mpi_test.count();
            times_mpi[i] = duration_mpi_long;
        }
    }

    if (rank == 0) {
        double sum_times_seq = array_sum_long(times_seq, test);
        double sum_times_pthread = array_sum_long(times_pthread, test);
        double sum_times_mpi = array_sum_long(times_mpi, test);

        printf("==============================================================\n");
        cout << "Time taken by multiple sequential: " << sum_times_seq / test << " microseconds" << endl;
        cout << "Time taken by multiple pthread: " << sum_times_pthread / test << " microseconds" << endl;
        cout << "Time taken by multiple mpi: " << sum_times_mpi / test << " microseconds" << endl;
        cout << "Performance boost by multiple pthreads: " << sum_times_seq / sum_times_pthread << endl;
        cout << "Performance boost by multiple mpi: " << sum_times_seq / sum_times_mpi << endl;
        printf("==============================================================\n");

        save_data(times_seq, times_pthread, times_mpi, threads_count, rows, test, save_path);
    }

    MPI_Finalize();
    return 0;
}

//
// Created by Alex_Nevskiy on 13.04.2023.
//

#include <iostream>
#include <Eigen/Dense>
#include <mpi.h>
#include "Matrix.h"
#include "utils.h"
#include "test.h"
#include "matrix_mul_pthreads.h"
#include "matrix_mul_mpi.h"

#define IS_TRUE(x) { if (!(x)) std::cout << __FUNCTION__ << " failed on line " << __LINE__ << std::endl; }

void test_square_matrix() {
    int size = 10;
    int lower_bound = 0;
    int upper_bound = 10000;
    double** mat1;
    double** mat2;

    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);

    Matrix matrix1(mat1, size, size);
    Matrix matrix2(mat2, size, size);

    Eigen::MatrixXd matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    Eigen::MatrixXd matrixXd2 = convert_to_eigen_matrix(mat2, size, size);

    Matrix matrix_result = matrix1 * matrix2;
    Eigen::MatrixXd matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    size = 100;
    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    matrixXd2 = convert_to_eigen_matrix(mat2, size, size);

    matrix_result = matrix1 * matrix2;
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    size = 300;
    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    matrixXd2 = convert_to_eigen_matrix(mat2, size, size);

    matrix_result = matrix1 * matrix2;
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    matrixXd2 = convert_to_eigen_matrix(mat1, size, size);

    matrix_result = matrix1 * matrix2;
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(!(matrix_result == matrixXd_result))
}

void test_random_size_matrix() {
    int rows1 = random_int(1, 10);
    int cols1_rows2 = random_int(1, 10);
    int cols2 = random_int(1, 10);
    int lower_bound = 0;
    int upper_bound = 10000;
    double** mat1;
    double** mat2;

    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);

    Matrix matrix1(mat1, rows1, cols1_rows2);
    Matrix matrix2(mat2, cols1_rows2, cols2);

    Eigen::MatrixXd matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
    Eigen::MatrixXd matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);

    Matrix matrix_result = matrix1 * matrix2;
    Eigen::MatrixXd matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    rows1 = random_int(1, 100);
    cols1_rows2 = random_int(1, 100);
    cols2 = random_int(1, 100);
    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
    matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);

    matrix_result = matrix1 * matrix2;
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    rows1 = random_int(1, 300);
    cols1_rows2 = random_int(1, 300);
    cols2 = random_int(1, 300);
    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
    matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);

    matrix_result = matrix1 * matrix2;
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)
}

void test_square_matrix_pthread(int threads_count) {
    int size = 100;
    int lower_bound = 0;
    int upper_bound = 10000;
    double** mat1;
    double** mat2;

    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);

    Matrix matrix1(mat1, size, size);
    Matrix matrix2(mat2, size, size);

    Eigen::MatrixXd matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    Eigen::MatrixXd matrixXd2 = convert_to_eigen_matrix(mat2, size, size);

    Matrix matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    Eigen::MatrixXd matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    size = 300;
    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    matrixXd2 = convert_to_eigen_matrix(mat2, size, size);

    matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    size = 500;
    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    matrixXd2 = convert_to_eigen_matrix(mat2, size, size);

    matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
    matrixXd2 = convert_to_eigen_matrix(mat1, size, size);

    matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(!(matrix_result == matrixXd_result))
}

void test_random_size_matrix_pthread(int threads_count) {
    int rows1 = random_int(1, 100);
    int cols1_rows2 = random_int(1, 100);
    int cols2 = random_int(1, 100);
    int lower_bound = 0;
    int upper_bound = 10000;
    double** mat1;
    double** mat2;

    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);

    Matrix matrix1(mat1, rows1, cols1_rows2);
    Matrix matrix2(mat2, cols1_rows2, cols2);

    Eigen::MatrixXd matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
    Eigen::MatrixXd matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);

    Matrix matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    Eigen::MatrixXd matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    rows1 = random_int(1, 300);
    cols1_rows2 = random_int(1, 300);
    cols2 = random_int(1, 300);
    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
    matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);

    matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)

    rows1 = random_int(1, 500);
    cols1_rows2 = random_int(1, 500);
    cols2 = random_int(1, 500);
    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
    generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
    matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);

    matrix_result = multiply_pthread(matrix1, matrix2, threads_count);
    matrixXd_result = matrixXd1 * matrixXd2;

    IS_TRUE(matrix_result == matrixXd_result)
}

void test_square_matrix_mpi(int rank, int num_processes) {
    int size = 100;
    int lower_bound = 0;
    int upper_bound = 10000;
    double** mat1;
    double** mat2;
    Matrix matrix1, matrix2, matrix_result;
    Eigen::MatrixXd matrixXd1, matrixXd2, matrixXd_result;

    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    if (rank == 0) {
        generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
        generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);
    }

    MPI_Bcast(&(mat1[0][0]), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    double** result_mpi;
    allocateSpace(&result_mpi, size, size);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, size, size);

        matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
        matrixXd2 = convert_to_eigen_matrix(mat2, size, size);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(matrix_result == matrixXd_result)
    }

    size = 300;
    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    if (rank == 0) {
        generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
        generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);
    }

    MPI_Bcast(&(mat1[0][0]), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    allocateSpace(&result_mpi, size, size);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, size, size);

        matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
        matrixXd2 = convert_to_eigen_matrix(mat2, size, size);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(matrix_result == matrixXd_result)
    }

    size = 500;
    allocateSpace(&mat1, size, size);
    allocateSpace(&mat2, size, size);

    if (rank == 0) {
        generate_double_matrix_array(mat1, size, size, lower_bound, upper_bound);
        generate_double_matrix_array(mat2, size, size, lower_bound, upper_bound);
    }

    MPI_Bcast(&(mat1[0][0]), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), size * size, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat2, size, size);

    allocateSpace(&result_mpi, size, size);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, size, size);

        matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
        matrixXd2 = convert_to_eigen_matrix(mat2, size, size);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(matrix_result == matrixXd_result)
    }

    matrix1 = Matrix(mat1, size, size);
    matrix2 = Matrix(mat1, size, size);

    allocateSpace(&result_mpi, size, size);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, size, size);

        matrixXd1 = convert_to_eigen_matrix(mat1, size, size);
        matrixXd2 = convert_to_eigen_matrix(mat2, size, size);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(!(matrix_result == matrixXd_result))
    }
}

void test_random_size_matrix_mpi(int rank, int num_processes) {
    int rows1;
    int cols1_rows2;
    int cols2;
    int lower_bound = 0;
    int upper_bound = 10000;
    double** mat1;
    double** mat2;
    Matrix matrix1, matrix2, matrix_result;
    Eigen::MatrixXd matrixXd1, matrixXd2, matrixXd_result;

    if (rank == 0) {
        rows1 = random_int(1, 100);
        cols1_rows2 = random_int(1, 100);
        cols2 = random_int(1, 100);
    }

    MPI_Bcast(&rows1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols1_rows2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    if (rank == 0) {
        generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
        generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);
    }

    MPI_Bcast(&(mat1[0][0]), rows1 * cols1_rows2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), cols1_rows2 * cols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    double** result_mpi;
    allocateSpace(&result_mpi, rows1, cols2);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, rows1, cols2);

        matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
        matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(matrix_result == matrixXd_result)
    }

    if (rank == 0) {
        rows1 = random_int(1, 300);
        cols1_rows2 = random_int(1, 300);
        cols2 = random_int(1, 300);
    }

    MPI_Bcast(&rows1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols1_rows2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    if (rank == 0) {
        generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
        generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);
    }

    MPI_Bcast(&(mat1[0][0]), rows1 * cols1_rows2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), cols1_rows2 * cols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    allocateSpace(&result_mpi, rows1, cols2);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, rows1, cols2);

        matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
        matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(matrix_result == matrixXd_result)
    }

    if (rank == 0) {
        rows1 = random_int(1, 500);
        cols1_rows2 = random_int(1, 500);
        cols2 = random_int(1, 500);
    }

    MPI_Bcast(&rows1, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols1_rows2, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&cols2, 1, MPI_INT, 0, MPI_COMM_WORLD);

    allocateSpace(&mat1, rows1, cols1_rows2);
    allocateSpace(&mat2, cols1_rows2, cols2);

    if (rank == 0) {
        generate_double_matrix_array(mat1, rows1, cols1_rows2, lower_bound, upper_bound);
        generate_double_matrix_array(mat2, cols1_rows2, cols2, lower_bound, upper_bound);
    }

    MPI_Bcast(&(mat1[0][0]), rows1 * cols1_rows2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(&(mat2[0][0]), cols1_rows2 * cols2, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    matrix1 = Matrix(mat1, rows1, cols1_rows2);
    matrix2 = Matrix(mat2, cols1_rows2, cols2);

    allocateSpace(&result_mpi, rows1, cols2);

    multiply_mpi(matrix1, matrix2, result_mpi, rank, num_processes);

    if (rank == 0) {
        matrix_result = Matrix(result_mpi, rows1, cols2);

        matrixXd1 = convert_to_eigen_matrix(mat1, rows1, cols1_rows2);
        matrixXd2 = convert_to_eigen_matrix(mat2, cols1_rows2, cols2);
        matrixXd_result = matrixXd1 * matrixXd2;

        IS_TRUE(matrix_result == matrixXd_result)
    }
}

Eigen::MatrixXd convert_to_eigen_matrix(double** data, int rows, int cols) {
    Eigen::MatrixXd matrixXd(rows, cols);
    for (int i = 0; i < rows; ++i)
        matrixXd.row(i) = Eigen::VectorXd::Map(&data[i][0], cols);
    return matrixXd;
}
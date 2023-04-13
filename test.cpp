//
// Created by Alex_Nevskiy on 13.04.2023.
//

#include <iostream>
#include <Eigen/Dense>
#include "Matrix.h"
#include "utils.h"
#include "test.h"

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

Eigen::MatrixXd convert_to_eigen_matrix(double** data, int rows, int cols) {
    Eigen::MatrixXd matrixXd(rows, cols);
    for (int i = 0; i < rows; ++i)
        matrixXd.row(i) = Eigen::VectorXd::Map(&data[i][0], cols);
    return matrixXd;
}
//
// Created by Alex_Nevskiy on 13.04.2023.
//

#ifndef PARALLELS_TEST_H
#define PARALLELS_TEST_H

#include <Eigen/Dense>

void test_square_matrix();
void test_random_size_matrix();
void test_square_matrix_pthread(int threads_count);
void test_random_size_matrix_pthread(int threads_count);
void test_square_matrix_mpi(int rank, int num_processes);
void test_random_size_matrix_mpi(int rank, int num_processes);
Eigen::MatrixXd convert_to_eigen_matrix(double** data, int rows, int cols);

#endif //PARALLELS_TEST_H

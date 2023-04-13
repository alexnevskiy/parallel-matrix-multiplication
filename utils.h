//
// Created by Alex_Nevskiy on 10.04.2023.
//

#ifndef PARALLELS_UTILS_H
#define PARALLELS_UTILS_H

#include "Matrix.h"
#include <chrono>

int allocateSpace(double*** array, int n, int m);
void print_matrix(Matrix& matrix);
double random_double(int lower_bound, int upper_bound);
int random_int(int lower_bound, int upper_bound);
void generate_int_matrix_array(double** mat, int rows, int cols, int lower_bound, int upper_bound);
void generate_double_matrix_array(double** mat, int rows, int cols, int lower_bound, int upper_bound);
long array_sum_long(long array[], int size);
void save_data(long* times_seq, long* times_pthread, long* times_mpi, int threads_count, int matrix_size,
               int experiments_count, char* save_path);

#endif //PARALLELS_UTILS_H

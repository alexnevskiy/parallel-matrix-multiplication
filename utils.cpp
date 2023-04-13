//
// Created by Alex_Nevskiy on 10.04.2023.
//

#include <cstdio>
#include "utils.h"
#include <filesystem>
#include <cstring>
#include <numeric>
#include <string>
#include <random>

int allocateSpace(double ***array, int n, int m) {
    /* allocate the n*m contiguous items */
    double *p = (double *) malloc(n * m * sizeof(double));
    if (!p) return -1;

    /* allocate the row pointers into the memory */
    (*array) = (double **) malloc(n * sizeof(double*));
    if (!(*array)) {
        free(p);
        return -1;
    }

    /* set up the pointers into the contiguous memory */
    for (int i=0; i<n; i++)
        (*array)[i] = &(p[i*m]);

    return 0;
}

void print_matrix(Matrix& matrix) {
    for (int i = 0; i < matrix.getRows(); i++) {
        for (int j = 0; j < matrix.getCols(); j++) {
            printf("%f ", matrix(i,j));
        }

        printf("\n");
    }
    printf("\n");
}

double random_double(int lower_bound, int upper_bound) {
    double d = (double)rand() / RAND_MAX;
    return lower_bound + d * (upper_bound - lower_bound);
}

int random_int(int lower_bound, int upper_bound) {
    return lower_bound + (rand() % static_cast<int>(upper_bound - lower_bound + 1));
}

void generate_int_matrix_array(double** mat, int rows, int cols, int lower_bound, int upper_bound) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = random_int(lower_bound, upper_bound);
        }
    }
}

void generate_double_matrix_array(double** mat, int rows, int cols, int lower_bound, int upper_bound) {
    std::uniform_real_distribution<double> unif(lower_bound,upper_bound);
    std::random_device dev;
    std::mt19937 re(dev());

    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i][j] = random_double(lower_bound, upper_bound);
        }
    }
}

long array_sum_long(long array[], int size) {
    long initial_sum = 0;
    return std::accumulate(array, array + size, initial_sum);
}

void save_data(long* times_seq, long* times_pthread, long* times_mpi, int threads_count, int matrix_size,
               int experiments_count, char* save_path) {
    std::string dir = std::string(save_path) + "\\data";
    std::string filename = dir + "\\data_" + std::to_string(threads_count) + "_" + std::to_string(matrix_size)
            + "_" + std::to_string(experiments_count) + ".txt";
    const int filename_length = filename.length();
    char* filename_char = new char[filename_length + 1];
    strcpy(filename_char, filename.c_str());

    std::filesystem::create_directory(dir);
    FILE *file = fopen(filename_char, "w");
    printf("\nSaving data");

    fprintf(file, "threads %d\n", threads_count);
    fprintf(file, "matrix size %d\n", matrix_size);
    fprintf(file, "number of experiments %d\n", experiments_count);

    fprintf(file, "algorithm sequentially\n");
    for (int i = 0; i < experiments_count; ++i) {
        fprintf(file, "%ld ", times_seq[i]);
    }

    fprintf(file, "\nalgorithm pthread\n");
    for (int i = 0; i < experiments_count; ++i) {
        fprintf(file, "%ld ", times_pthread[i]);
    }

    fprintf(file, "\nalgorithm mpi\n");
    for (int i = 0; i < experiments_count; ++i) {
        fprintf(file, "%ld ", times_mpi[i]);
    }
    fclose(file);
}
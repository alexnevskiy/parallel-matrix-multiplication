//
// Created by Alex_Nevskiy on 07.04.2023.
//
#include "matrix_mul_pthreads.h"
#include <pthread.h>
#include <cstdio>
#include "Matrix.h"

//pthread_mutex_t th_mutex;

struct arg_struct {
    arg_struct(int start_row, int end_row, int cols, Matrix& matrix1, Matrix& matrix2, double** result) :
            start_row(start_row), end_row(end_row), cols(cols), matrix1(matrix1), matrix2(matrix2), result(result) {}
    int start_row;
    int end_row;
    int cols;
    Matrix& matrix1;
    Matrix& matrix2;
    double** result;
};

void* compute_rows(void* args) {
    auto* arguments = (arg_struct*) args;
    int matrix1_cols = arguments->matrix1.getCols();

    for (int i = arguments->start_row; i < arguments->end_row; i++) {
        for (int j = 0; j < arguments->cols; j++) {
            double sum = 0;
            for (int k = 0; k < matrix1_cols; k++) {
                sum += (arguments->matrix1(i, k) * arguments->matrix2(k, j));
            }
//            pthread_mutex_lock(&th_mutex);
            arguments->result[i][j] = sum;
//            pthread_mutex_unlock(&th_mutex);
        }
    }
    pthread_exit(nullptr);
}

Matrix multiply_pthread(Matrix& matrix1, Matrix& matrix2, int threads_count) {
    pthread_attr_t attr;
    pthread_t threads_pool[threads_count];
    double** result;

    int rc = pthread_attr_init(&attr);
    if (rc == -1) {
        perror("Error: pthread_attr_init failed\n");
        exit(1);
    }

    rc = pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);
    if (rc == -1) {
        perror("Error: pthread_attr_setdetachstate failed\n");
        exit(2);
    }

    int result_rows = matrix1.getRows();
    int result_cols = matrix2.getCols();
    result = new double*[result_rows];
    for (int i = 0; i < result_rows; ++i) {
        result[i] = new double[result_cols];
    }

    int n_used_threads = result_rows < threads_count ? result_rows : threads_count;
    int n_work_rows = result_rows < threads_count ? 1 : result_rows / threads_count;
    int n_rest_rows = result_rows < threads_count ? 0 : result_rows % threads_count;
    int start_row;
    int end_row = 0;

    for (int i = 0; i < n_used_threads; i++) {
        int n_work_rows_curr = n_work_rows;
        if (n_rest_rows != 0) {
            n_work_rows_curr++;
            n_rest_rows--;
        }
        start_row = end_row;
        end_row += n_work_rows_curr;

        auto *args = new arg_struct(start_row, end_row, result_cols, matrix1, matrix2, result);
        int ret = pthread_create(&threads_pool[i], &attr, compute_rows, args);
        if (ret != 0) {
            printf("Error: pthread_create() failed\n");
            exit(3);
        }
    }

    for (int i = 0; i < n_used_threads; i++) {
        pthread_join(threads_pool[i], nullptr);
    }
    pthread_attr_destroy(&attr);

    Matrix result_matrix(result, result_rows, result_cols);
    return result_matrix;
}
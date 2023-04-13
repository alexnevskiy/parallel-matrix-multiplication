//
// Created by Alex_Nevskiy on 08.04.2023.
//
#include "matrix_mul_mpi.h"
#include <mpi.h>
#include <cstdlib>
#include "Matrix.h"

#define START_ROW_TAG 0
#define END_ROW_TAG 1
#define RESULT_TAG 2

void compute_rows(int start_row, int end_row, int cols, Matrix& matrix1, Matrix& matrix2, double** result) {
    int matrix1_cols = matrix1.getCols();

    for (int i = start_row; i < end_row; i++) {
        for (int j = 0; j < cols; j++) {
            double sum = 0;
            for (int k = 0; k < matrix1_cols; k++) {
                sum += (matrix1(i, k) * matrix2(k, j));
            }
            result[i][j] = sum;
        }
    }
}

void multiply_mpi(Matrix& matrix1, Matrix& matrix2, double** result, int rank, int threads_count) {
    MPI_Status status;
    MPI_Request request;

    int result_rows = matrix1.getRows();
    int result_cols = matrix2.getCols();
    int start_row;
    int end_row = 0;
    int n_used_threads;

    if (rank == 0) {
        n_used_threads = result_rows < threads_count ? result_rows : threads_count;
        int n_work_rows = result_rows < threads_count ? 1 : result_rows / threads_count;
        int n_rest_rows = result_rows < threads_count ? 0 : result_rows % threads_count;

        int n_work_rows_curr = n_work_rows;
        if (n_rest_rows != 0) {
            n_work_rows_curr++;
            n_rest_rows--;
        }
        start_row = end_row;
        end_row += n_work_rows_curr;
        int start_row_main = start_row;
        int end_row_main = end_row;

        for (int i = 1; i < n_used_threads; i++) {
            n_work_rows_curr = n_work_rows;
            if (n_rest_rows != 0) {
                n_work_rows_curr++;
                n_rest_rows--;
            }
            start_row = end_row;
            end_row += n_work_rows_curr;

            MPI_Isend(&start_row, 1, MPI_INT, i, START_ROW_TAG, MPI_COMM_WORLD, &request);
            MPI_Isend(&end_row, 1, MPI_INT, i, END_ROW_TAG, MPI_COMM_WORLD, &request);
        }
        start_row = start_row_main;
        end_row = end_row_main;
    } else {
        MPI_Recv(&start_row, 1, MPI_INT, 0, START_ROW_TAG, MPI_COMM_WORLD, &status);
        MPI_Recv(&end_row, 1, MPI_INT, 0, END_ROW_TAG, MPI_COMM_WORLD, &status);
    }

    compute_rows(start_row, end_row, result_cols, matrix1, matrix2, result);

    if (rank == 0) {
        for (int i = 1; i < n_used_threads; i++) {
            MPI_Recv(&start_row, 1, MPI_INT, i, START_ROW_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&end_row, 1, MPI_INT, i, END_ROW_TAG, MPI_COMM_WORLD, &status);
            MPI_Recv(&result[start_row][0], (end_row - start_row) * result_cols, MPI_DOUBLE, i, RESULT_TAG, MPI_COMM_WORLD, &status);
        }
    } else {
        MPI_Isend(&start_row, 1, MPI_INT, 0, START_ROW_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&end_row, 1, MPI_INT, 0, END_ROW_TAG, MPI_COMM_WORLD, &request);
        MPI_Isend(&result[start_row][0], (end_row - start_row) * result_cols, MPI_DOUBLE, 0, RESULT_TAG, MPI_COMM_WORLD, &request);
    }
}
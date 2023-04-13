//
// Created by Alex_Nevskiy on 08.04.2023.
//

#ifndef PARALLELS_MATRIX_MUL_MPI_H
#define PARALLELS_MATRIX_MUL_MPI_H

#include "Matrix.h"

void* compute_rows(void* args);
void multiply_mpi(Matrix& matrix1, Matrix& matrix2, double** result, int rank, int threads_count);

#endif //PARALLELS_MATRIX_MUL_MPI_H

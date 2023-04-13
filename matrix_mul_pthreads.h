//
// Created by Alex_Nevskiy on 08.04.2023.
//

#ifndef PARALLELS_MATRIX_MUL_PTHREADS_H
#define PARALLELS_MATRIX_MUL_PTHREADS_H

#include "Matrix.h"

void* compute_rows(void* args);
Matrix multiply_pthread(Matrix& mat, Matrix& matrix2, int threads_count);

#endif //PARALLELS_MATRIX_MUL_PTHREADS_H

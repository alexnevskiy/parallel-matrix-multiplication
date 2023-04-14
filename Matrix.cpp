//
// Created by Alex_Nevskiy on 06.04.2023.
//

#include <stdexcept>
#include "Matrix.h"
#define EPSILON (1.0E-4)

Matrix::Matrix(int rows, int cols) : rows(rows), cols(cols) {
    allocateSpace();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = 0;
        }
    }
}

Matrix::Matrix(double** a, int rows, int cols) : rows(rows), cols(cols) {
    allocateSpace();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = a[i][j];
        }
    }
}

Matrix::Matrix() : rows(1), cols(1) {
    allocateSpace();
    mat[0][0] = 0;
}

Matrix::Matrix(const Matrix& m) : rows(m.rows), cols(m.cols) {
    allocateSpace();
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            mat[i][j] = m.mat[i][j];
        }
    }
}

void Matrix::allocateSpace() {
    mat = new double*[rows];
    for (int i = 0; i < rows; ++i) {
        mat[i] = new double[cols];
    }
}

Matrix &Matrix::operator*=(const Matrix& m) {
    if (this->cols != m.rows) {
        throw std::invalid_argument("The number of columns of matrix 1 is not equal to the number of rows of matrix 2.");
    }

    Matrix temp(rows, m.cols);
    for (int i = 0; i < temp.rows; ++i) {
        for (int j = 0; j < temp.cols; ++j) {
            double sum = 0;
            for (int k = 0; k < cols; ++k) {
                sum += (mat[i][k] * m.mat[k][j]);
            }
            temp.mat[i][j] = sum;
        }
    }
    return (*this = temp);
}

bool Matrix::operator==(const Matrix& m) const {
    if ((rows != m.rows) or (cols != m.cols)) {
        return false;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!doubleEquals(mat[i][j], m.mat[i][j])) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::operator==(const Eigen::MatrixXd& m) const {
    if ((rows != m.rows()) or (cols != m.cols())) {
        return false;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            if (!doubleEquals(mat[i][j], m(i, j))) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix::doubleEquals(double a, double b) {
    return fabs(a - b) < EPSILON;
}

Matrix operator*(const Matrix& m1, const Matrix& m2) {
    Matrix temp(m1);
    return (temp *= m2);
}

//
// Created by Alex_Nevskiy on 06.04.2023.
//

#ifndef PARALLELS_MATRIX_H
#define PARALLELS_MATRIX_H

#include <Eigen/Dense>


class Matrix {
public:
    Matrix(int, int);
    Matrix(double**, int, int);
    Matrix();
    Matrix(const Matrix&);

    inline double& operator()(int x, int y) { return mat[x][y]; }

    Matrix& operator*=(const Matrix&);
    bool operator==(const Matrix&) const;
    bool operator==(const Eigen::MatrixXd&) const;

    int getRows() const { return rows; };
    int getCols() const { return cols; };
    double** getMatrix() const { return mat; };

private:
    int rows, cols;
    double **mat{};

    void allocateSpace();
    static bool doubleEquals(double a, double b) ;
};

Matrix operator*(const Matrix&, const Matrix&);

#endif //PARALLELS_MATRIX_H

// Author Johann Fredrik Jadebeck
//
// C++ port of quantum annealing python script.
#ifndef QUANTUMANNEALING_H
#define QUANTUMANNEALING_H

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

#include "quantumMechanics.hpp"

using namespace Eigen;

template<typename T>
std::vector<double> linspace(const T& s, const T& e, const int& n);
std::vector<std::vector<double>> create_const(const int& x, const int& y, const double& c);
std::set<std::string> zeeman(const int& N, const double& m);
int cardaniac(const double& A, const double& B, const double& C, const double& D);

Matrix<double, Dynamic, Dynamic> H0(const int& N, const int& p);
Matrix<double, Dynamic, Dynamic> Vtf(const int& N);
Matrix<double, Dynamic, Dynamic> Vaff(const int& N);
Matrix<double, Dynamic, Dynamic> H0plusVtf(const int& N, const double& s, const int& p);
Matrix<double, Dynamic, Dynamic> H0plusVaffplusVtf(const int& N, const double& s, const double& l, const int& p);

SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic>> diagonalize(Matrix<double, Dynamic, Dynamic> H);
void lambdaOne(const int& p);
#endif

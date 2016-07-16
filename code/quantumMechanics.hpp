// Author Fredrik Jadebeck
//
// Headerfile for the quantumMechanics.py port
#ifndef QUANTUMMECHANICS_H
#define QUANTUMMECHANICS_H

#include <Eigen/Core>
#include <set>
#include <vector>

using namespace Eigen;


std::vector<double> gen_m(const int& N);

int kronecker(const double& m, const double& n);

double mSxn(const double& m, const double& n, const double& S);

double mSzn(const double& m, const double& n, const double& S);

Matrix<double, Dynamic, Dynamic> Sx(const int& N);

Matrix<double, Dynamic, Dynamic> Sz(const int& N);


#endif

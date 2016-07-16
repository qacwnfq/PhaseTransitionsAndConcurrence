// Author Fredrik Jadebeck
//
// Headerfile for the quantumMechanics.py port
#ifndef QUANTUMMECHANICS_H
#define QUANTUMMECHANICS_H

#include <Eigen/Core>
#include <set>
#include <vector>


std::vector<double> gen_m(const int& N);
int kronecker(const double& m, const double& n);
double mSxn(const double& m, const double& n, const double& S);
double mSzn(const double& m, const double& n, const double& S);
// auto Sx(const int& N);
// auto Sz(const int& N);
#endif

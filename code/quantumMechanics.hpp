// Author Fredrik Jadebeck
//
// Headerfile for the quantumMechanics.py port
#ifndef QUANTUMMECHANICS_H
#define QUANTUMMECHANICS_H

#include <Eigen/Core>
#include <map>
#include <string>
#include <vector>

using namespace Eigen;


std::vector<double> gen_m(const int& N);
int kronecker(const double& m, const double& n);
double mSxn(const double& m, const double& n, const double& S);
double mSzn(const double& m, const double& n, const double& S);
Matrix<double, Dynamic, Dynamic> Sx(const int& N);
Matrix<double, Dynamic, Dynamic> Sz(const int& N);

class state
{
public:
  std::vector<std::string> quantumState;
  double norm;
  
  state(std::string spins);
  double getNorm();
  double scalar_product(const state& other);
  double tensor_product(const state& other);
  state operator+(const state& other);
};

class ketBra
{
public:
  double amplitude;
  state ket;
  state bra;
  
  ketBra(state ket, state bra);

  ketBra multiply_with(ketBra other);
};

class dm
{
public:
  std::map<ketBra, double> rho;
  int N;
  int S;
  std::vector<double> ms;
  std::vector<state> z;

  dm(Matrix<double, Dynamic, Dynamic>, int N);
  Matrix<double, Dynamic, Dynamic> nparray();
  dm ptrace(int k);
};

void zeeman();

#endif

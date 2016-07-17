// Author Fredrik Jadebeck
//
// Headerfile for the quantumMechanics.py port
#ifndef QUANTUMMECHANICS_H
#define QUANTUMMECHANICS_H

#include <Eigen/Core>
#include <map>
#include <iostream>
#include <string>
#include <vector>

using namespace Eigen;

std::vector<double> gen_m(const int& N);
int kronecker(const double& m, const double& n);
double mSxn(const double& m, const double& n, const double& S);
double mSzn(const double& m, const double& n, const double& S);
Matrix<double, Dynamic, Dynamic> Sx(const int& N);
Matrix<double, Dynamic, Dynamic> Sz(const int& N);

//class prototypes
class state;
class ketBra;
class dm;

//class definitions
class state
{
public:
  std::vector<std::string> quantumState;

  state();
  state(std::string spins);
  state(std::vector<std::string> spins);
  state(const state& obj);
  double getNorm() const;
  double scalar_product(const state& other) const;
  ketBra tensor_product(const state& other) const;
  state operator+(const state& other) const;
  bool operator==(const state& other) const;
};
// this should be declared as a free function!
std::ostream& operator<<(std::ostream& os, const state& other);

class ketBra
{
public:
  double amplitude;
  state tket;
  state tbra;

  ketBra(const state& ket, const state& bra);
  ketBra(const ketBra& obj);
  ketBra multiply_with(const ketBra& other);
};
std::ostream& operator<<(std::ostream& os, const ketBra& other);

class dm
{
public:
  std::map<ketBra, double> rho;
  int N;
  int S;
  std::vector<double> ms;
  std::vector<state> z;

  dm(Matrix<double, Dynamic, Dynamic> rho,
     std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis,
     const int& N);
  dm(const dm& obj);
  Matrix<double, Dynamic, Dynamic> nparray();
  dm ptrace(const int& k);
};

void zeeman();

#endif

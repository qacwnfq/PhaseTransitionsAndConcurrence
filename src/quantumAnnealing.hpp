// Author Johann Fredrik Jadebeck
//
// C++ port of quantum annealing python script.

#include <complex>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

#ifndef QUANTUMANNEALING_H
#define QUANTUMANNEALING_H

typedef std::complex<double> complex;
typedef long double BigDouble;

const std::complex<double> I = {0.0, 1.0};

template<typename T> std::vector<double> linspace(const T& s, const T& e, const int& n)
{
  // This is a useful template to recreate numpys linspace from python
  double start = static_cast<double>(s);
  double end = static_cast<double>(e);
  double num = static_cast<double>(n);
  double delta = (end - start)/(num-1);

  std::vector<double> linspaced(num);
  for(int i=0; i<num; ++i)
  {
    linspaced[i] = start + delta*i;
  }
  linspaced[end];
  return linspaced;
}

std::vector<double> gen_m(const int& N);
int kronecker(const double& m, const double& n);
double mSxn(const double& m, const double& n, const double& S);
complex mSyn(const double& m, const double& n, const double& S);
double mSzn(const double& m, const double& n, const double& S);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Sx(const int& N);
Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> Sy(const int& N);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Sz(const int& N);
Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> dM2c(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> m);
complex expectationValue(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state, Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> op);
complex vplus(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
complex vminus(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
complex xplus(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
complex xminus(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
complex w(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
complex y(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
complex u(const int& N, Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
Eigen::Matrix<complex, Eigen::Dynamic, Eigen::Dynamic> twoSystemRho(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
double altConcurrence(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> state);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ptrace(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& rho, std::vector<std::vector<BigDouble> > pascal, const int& N);
std::vector<std::vector<BigDouble> > pascalTriangle(const int& prev, const int& N);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H0(const int& N, const int& p);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Vtf(const int& N);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> Vaff(const int& N);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H0plusVtf(const int& N, const double& s, const int& p);
Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> H0plusVaffplusVtf(const int& N, const double& s, const double& l, const int& p);

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> ket2dm(Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>> es, const int& N);

double concurrence(Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> rho);
double calculateConcurrence(Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,
			    Eigen::Dynamic, Eigen::Dynamic> > es,
			    std::vector<std::vector<BigDouble> > pascal,
			    const int& N);


std::vector<std::vector<double>> lambdaOne(const int& p, const std::vector<double>& s_list, const std::vector<int>& N_list);
std::vector<std::vector<double>> lambdaOneConcurrence(const int& p, const std::vector<double>& s_list,  const std::vector<int>& N_list);
std::vector<std::vector<double>> lambdaNotOne(const double& l, const int& p, const std::vector<double>& s_list,  const std::vector<int>& N_list);
std::vector<std::vector<double>> lambdaNotOneConcurrence(const double& l, const int& p, const std::vector<double>& s_list,  const std::vector<int>& N_list);
#endif

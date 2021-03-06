// Author Johann Fredrik Jadebeck
//
// C++ port of quantum annealing python script.

#include "quantumAnnealing.hpp"

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unsupported/Eigen/MatrixFunctions>
#include <vector>


using namespace Eigen;

std::vector<double> gen_m(const int& N)
{
  assert(N > 0);
  double S = (double)N/2.;
  std::vector<double> m;
  for(int i=0; i<N+1; ++i)
  {
    m.push_back(-S+i);
  }
  return m;
}

int kronecker(const double& m, const double& n)
{
  if(m==n)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

double mSxn(const double& m, const double& n, const double& S)
{
  assert(S > 0);
  return (kronecker(m, n+1) + kronecker(m+1, n))*0.5*std::sqrt(S*(S+1) -m*n);
}

complex mSyn(const double& m, const double& n, const double& S)
{
  assert(S > 0);
  return (kronecker(m, n+1) - kronecker(m+1, n))*0.5/I*std::sqrt(S*(S+1) -m*n);
}

double mSzn(const double& m, const double& n, const double& S)
{
  assert(S > 0);
  return kronecker(m, n)*m;
}

Matrix<double, Dynamic, Dynamic> Sx(const int& N)
{
  assert(N > 0);
  double S = (double)N/2;
  std::vector<double> m = gen_m(N);
  Matrix<double, Dynamic, Dynamic> Sx;
  Sx.resize(N+1, N+1);
  Sx.setZero();
  for(int i=0; i<N; ++i)
  {
    Sx(i, i+1) = mSxn(m[i+1], m[i], S);
    Sx(i+1, i) = mSxn(m[i], m[i+1], S);
  }
  return Sx;
}

Matrix<complex, Dynamic, Dynamic> Sy(const int& N)
{
  assert(N > 0);
  double S = (double)N/2;
  std::vector<double> m = gen_m(N);
  Matrix<complex, Dynamic, Dynamic> Sy;
  Sy.resize(N+1, N+1);
  Sy.setZero();
  for(int i=0; i<N; ++i)
  {
    Sy(i, i+1) = mSyn(m[i+1], m[i], S);
    Sy(i+1, i) = mSyn(m[i], m[i+1], S);
  }
  return Sy;
}

Matrix<double, Dynamic, Dynamic> Sz(const int& N)
{
  assert(N > 0);
  double S = (double)N/2;
  std::vector<double> m = gen_m(N);
  Matrix<double, Dynamic, Dynamic> Sz;
  Sz.resize(N+1, N+1);
  Sz.setZero();
  for(int i=0; i<N+1; ++i)
  {
    Sz(i, i) = mSzn(m[i], m[i], S);
  }
  return Sz;
}

Matrix<complex, Dynamic, Dynamic> dM2c(Matrix<double, Dynamic, Dynamic> m)
{
  Matrix<complex, Dynamic, Dynamic> ret;
  ret.resize(m.rows(), m.cols());
  ret.setZero();
  for(int i=0; i<m.rows(); ++i)
    for(int j=0; j<m.cols(); ++j)
    {
      std::complex<double> c = m(i, j);
      ret(i, j) = c;
    }
  return ret;
}

Matrix<double, Dynamic, Dynamic> cM2d(Matrix<complex, Dynamic, Dynamic> m)
{
  Matrix<double, Dynamic, Dynamic> ret;
  ret.resize(m.rows(), m.cols());
  ret.setZero();
  for(int i=0; i<m.rows(); ++i)
    for(int j=0; j<m.cols(); ++j)
    {
      double c = std::real(m(i, j));
      ret(i, j) = c;
    }
  return ret;
}


complex expectationValue(Matrix<double, Dynamic, Dynamic> state, Matrix<complex, Dynamic, Dynamic> op)
{
  complex exp = 0;
  for(int i=0; i<op.rows(); ++i)
  {
    for(int j=0; j<op.rows(); ++j)
    {
      exp += state(i, 0)*op(i, j)*state(j, 0);
    }
  }
  return exp;
}

complex vplus(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  complex v = 0;
  v = ((double)N*N);
  v -= (double)2*N;
  auto a = dM2c(Sz(N)*Sz(N));
  v += 4.*expectationValue(state, a);
  v += 4.*expectationValue(state, dM2c(Sz(N)))*(double)(N-1);
  v /= 4*N*(N-1);
  return v;
}

complex vminus(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  complex v = ((double)N*N-2.*N+4.*expectationValue(state, dM2c(Sz(N)*Sz(N))) -
	       4.*expectationValue(state, dM2c(Sz(N)))*(double)(N-1));
  v /= 4*N*(N-1);
  return v;
}

complex xplus(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  Matrix<complex, Dynamic, Dynamic> Splus = dM2c(Sx(N)) + I*Sy(N);
  complex x = (double)(N-1)*expectationValue(state, Splus) + expectationValue(state, Splus*Sz(N) + Sz(N)*Splus);
  x /= (double)(2.*N*(N-1));
  return x;
}

complex xminus(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  Matrix<complex, Dynamic, Dynamic> Splus = dM2c(Sx(N)) + I*Sy(N);
  complex x = (double)(N-1)*expectationValue(state, Splus) - expectationValue(state, Splus*Sz(N) + Sz(N)*Splus);
  x /= (double)(2*N*(N-1));
  return x;
}

complex w(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  complex w = ((double)N*N-4.*expectationValue(state, dM2c(Sz(N)*Sz(N))));
  w /= (double)(4*N*(N-1));
  return w;
}

complex y(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  complex y = 2.*expectationValue(state, dM2c(Sx(N)*Sx(N)) + Sy(N)*Sy(N)) - (double)N;
  y /= (double)2*N*(N-1);
  return y;
}

complex u(const int& N, Matrix<double, Dynamic, Dynamic> state)
{
  Matrix<complex, Dynamic, Dynamic> Splus = dM2c(Sx(N)) + I*Sy(N);
  complex u = expectationValue(state, Splus*Splus);
  u /= (double)N*(N-1);
  return u;
}

Matrix<complex, Dynamic, Dynamic> twoSystemRho(Matrix<double, Dynamic, Dynamic> state)
{
  Matrix<complex, Dynamic, Dynamic> rho;
  rho.resize(4, 4);
  rho.setZero();
  const int N = state.rows() - 1;
  rho(0, 0) = vplus(N, state);
  rho(0, 1) = std::conj(xplus(N, state));
  rho(0, 2) = std::conj(xplus(N, state));
  rho(0, 3) = std::conj(u(N, state));
  rho(1, 0) = xplus(N, state);
  rho(1, 1) = w(N, state);
  rho(1, 2) = std::conj(y(N, state));
  rho(1, 3) = std::conj(xminus(N, state));
  rho(2, 0) = xplus(N, state);
  rho(2, 1) = y(N, state);
  rho(2, 2) = w(N, state);
  rho(2, 3) = std::conj(xminus(N, state));
  rho(3, 0) = u(N, state);
  rho(3, 1) = xminus(N, state);
  rho(3, 2) = xminus(N, state);
  rho(3, 3) = vminus(N, state);
  return rho;
}

double altConcurrence(Matrix<double, Dynamic, Dynamic> state)
{
  Matrix<double, Dynamic, Dynamic> rho = cM2d(twoSystemRho(state));
  Matrix<double, Dynamic, Dynamic> ssy;
  ssy.resize(4, 4);
  ssy.setZero();
  ssy(0, 3) = -1;
  ssy(1, 2) = 1;
  ssy(2, 1) = 1;
  ssy(3, 0) = -1;
  Matrix<double, Dynamic, Dynamic> R = rho*ssy;
  R = R*R;
  // Uses the regular eigensolver here,
  // because the R matrix is not selfadjoint.
  EigenSolver<Matrix<double, Dynamic, Dynamic> > es;
  es.compute(R);
  auto ev = es.eigenvalues();
  // Applies std::abs() before taking the sqrt to avoid
  // complex numbers. When the eigenvalues are close to 0
  // the floating point precision will sometimes lead to
  // negative values, while they should actually be zero.
  std::vector<double> lambdas = {std::sqrt(std::abs(ev(0))),
  				 std::sqrt(std::abs(ev(1))),
  				 std::sqrt(std::abs(ev(2))),
  				 std::sqrt(std::abs(ev(3)))};
  std::sort(lambdas.begin(), lambdas.end());
  double c = std::max(0., lambdas[3] - lambdas[2] - lambdas[1] - lambdas[0]);
  return c;
}

Matrix<double, Dynamic, Dynamic> ptrace(const Matrix<double, Dynamic, Dynamic>& rho, std::vector<std::vector<BigDouble> > pascal, const int& N)
{
  //N is equal to number of spins+1
  Matrix<double, Dynamic, Dynamic> res;
  res.resize(N-1, N-1);
  res.setZero();
  std::vector<BigDouble> New = pascal[N-2];
  std::vector<BigDouble> Old = pascal[N-1];
  double temp = 0;
  for(int i=0; i<N; ++i)
  {
    for(int j=0; j<N; ++j)
    {
      double factor = (Old[i]);
      factor*= (Old[j]);
      if((i < N-1) and (j < N-1))
      {
      	temp = rho(i, j) / factor;
      	temp *= (New[i]);
      	temp *= (New[j]);
	res(i, j) += temp;
      }
      if((i > 0) and (j > 0))
      {
      	temp = rho(i, j) / factor;
      	temp *= (New[i-1]);
      	temp *= (New[j-1]);
	res(i-1, j-1) += temp;
      }
    }
  }
  return res;
}

std::vector<std::vector<BigDouble> > pascalTriangle(const int& prev, const int& N)
{
    // Calculates pascal triangle up to N spins starting from line prev which means N+1 lines
  // in O(N^2). Be careful of integerowerflow though
  std::vector<std::vector<BigDouble> > triangle;
  // Starts at line 0 even if its not necessary because
  // this leads to the vector index being equal to the
  // number of spins.
  for(int line=prev; line<N+1; line++)
  {
    long double C = 1;
    std::vector<BigDouble> lin;
    for(int i=1; i<line+2; i++)
    {
      lin.push_back(C);
      C = C*(line -i + 1)/i;
    }
    for(auto k : lin)
    {
    }
    for(int i=0; i<lin.size(); ++i)
    {
      lin[i] = std::sqrt(lin[i]);
    }
    triangle.push_back(lin);
  }
  return triangle;
}

std::vector<double> readLimit(std::string title)
{
  // Reads the limit for concurrence
  // which the python script quantumAnnealing.py
  // calculates into a vector.
  std::ifstream file;
  file.open(title);
  std::vector<double> result;
  std::string line;

  double c;
  // Skips header line
  getline(file, line);
  while(std::getline(file, line))
  {
    // Reads line
    std::stringstream  lineStream(line);
    std::string        cell;
    while(std::getline(lineStream, cell, ','))
    {
      // Reads every cell in a line.
      // The concurrence values are
      // stored in the last cell.
      c = std::stod(cell);
    }
    result.push_back(c);
  }
  file.close();
  return result;
}


Matrix<double, Dynamic, Dynamic> H0(const int& N, const int& p)
{
  // Returns matrix representation of the target Hamiltonian H0
  double S = double(N)/2.;
  Matrix<double, Dynamic, Dynamic> H0 = Sz(N);
  H0 /= S;
  Matrix<double, Dynamic, Dynamic> Temp(H0);
  for(int i=0; i<p-1; ++i){
    H0 = H0*Temp;
  }
  H0 *= -N;
  return H0;
}

Matrix<double, Dynamic, Dynamic> Vtf(const int& N)
{
  // Returns matrix representation of the transverse field Vtf
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> Vtf = Sx(N);
  Vtf /= S;
  Vtf *= -N;
  return Vtf;
}

Matrix<double, Dynamic, Dynamic> Vaff(const int& N)
{
  // Returns matrix representation of the antiferromagnetic term Vaff
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> Vaff = Sx(N);
  Vaff /= S;
  Vaff *= Vaff;
  Vaff *= N;
  return Vaff;
}

Matrix<double, Dynamic, Dynamic> H0plusVtf(const int& N, const double& s, const int& p)
{
  return s*H0(N, p) + (1-s)*Vtf(N);
}

Matrix<double, Dynamic, Dynamic> H0plusVaffplusVtf(const int& N, const double& s, const double& l, const int& p)
{
  return s*l*H0(N, p) + s*(1-l)*Vaff(N) + (1-s)*Vtf(N);
}

Matrix<double, Dynamic, Dynamic> ket2dm(SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es, const int& N)
{
  return es.eigenvectors().col(0)*es.eigenvectors().col(0).transpose();
}

Matrix<double, Dynamic, Dynamic> extract2particleDm(Matrix<double, Dynamic, Dynamic> rho,
						 std::vector<std::vector<BigDouble> > pascal,
						 const int& N)
{
  // Takes a density matrix in zee man basis and
  // traces out all particles expect for 2.
  int SpinsPlusOne = N+1;
  for(int i=0; i<N-2; ++i)
  {
    // ptrace() expects spins+1 to know
    // the size of rho ((N+1)x(N+1))
    rho = ptrace(rho, pascal, SpinsPlusOne);
    SpinsPlusOne--;
  }
  return rho;
}


double concurrence(Matrix<double, Dynamic, Dynamic> rho)
{
  // Calculates concurrence for a 2 particle
  // density matrix in the zeeman basis.
  Matrix<double, Dynamic, Dynamic> copy = rho;
  rho.resize(4, 4);
  rho.setZero();
  // Applies the sigmay_i tensor sigmay_j to rho
  rho(0, 0) = -copy(0, 2);
  rho(1, 0) = -copy(1, 2);
  rho(2, 0) = -copy(2, 2);
  rho(0, 1) = copy(0, 1);
  rho(1, 1) = copy(1, 1);
  rho(2, 1) = copy(2, 1);
  rho(0, 2) = -copy(0, 0);
  rho(1, 2) = -copy(1, 0);
  rho(2, 2) = -copy(2, 0);

  // Calculates R matrix and store it in rho variable.
  rho = rho*rho;
  // Uses the regular eigensolver here,
  // because the R matrix is not selfadjoint.
  EigenSolver<Matrix<double, Dynamic, Dynamic> > es;
  es.compute(rho);
  auto ev = es.eigenvalues();
  // Applies std::abs() before taking the sqrt to avoid
  // complex numbers. When the eigenvalues are close to 0
  // the floating point precision will sometimes lead to
  // negative values, while they should actually be zero.
  std::vector<double> lambdas = {std::sqrt(std::abs(ev(0))),
				 std::sqrt(std::abs(ev(1))),
				 std::sqrt(std::abs(ev(2))),
				 std::sqrt(std::abs(ev(3)))};
  std::sort(lambdas.begin(), lambdas.end());
  double c = std::max(0., lambdas[3] - lambdas[2] - lambdas[1] - lambdas[0]);
  return c;
}

double calculateConcurrence(SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es,
			    std::vector<std::vector<BigDouble> > pascal,
			    const int& N)
{
  Matrix<double, Dynamic, Dynamic> rho = ket2dm(es, N);
  rho = extract2particleDm(rho, pascal, N);
  double trace = rho.trace();
  // This assert protects from integer overflow.
  assert(std::abs(trace-1.) < 0.0001);
  double c = concurrence(rho);
  return c;
}

std::vector<std::vector<double>> lambdaOne(const int& p, const std::vector<double>& s_list, const std::vector<int>& N_list)
{
  // Calculates the groundstate energy for lambda=1
  std::vector<std::vector<double>> energies;
  for(int N: N_list)
  {
    std::cout << "Calculating " << N << " Spins." << std::endl;
    std::vector<double> energy;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es.compute(H0plusVtf(N, s, p));
      double ev = es.eigenvalues()(0);
      energy.push_back(ev/N);
    }
    energies.push_back(energy);
  }
  std::cout << "Done." << std::endl;
  return energies;
}

std::vector<std::vector<double>> lambdaOneConcurrence(const int& p, const std::vector<double>& s_list,  const std::vector<int>& N_list)
{
  // Calculates the rescaled concurrence for lambda=1
  std::vector<std::vector<double> > concurrences;
  std::vector<std::vector<BigDouble> > pascal, temp;
  for(int N: N_list)
  {
    temp = pascalTriangle(pascal.size(), N);
    pascal.insert(pascal.end(), temp.begin(), temp.end());
    std::cout << "Calculating concurrence " << N << " Spins." << std::endl;
    std::vector<double> concurrence;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es.compute(H0plusVtf(N, s, p));
      concurrence.push_back(calculateConcurrence(es, pascal, N)*(N-1));
    }
    concurrences.push_back(concurrence);
  }
  std::cout << "Done." << std::endl;
  return concurrences;
}

std::vector<std::vector<double>> lambdaNotOne(const double& l, const int& p, const std::vector<double>& s_list,  const std::vector<int>& N_list)
{  
  // Calculates the groundstate energy for lambda=l
  assert(l >= 0);
  assert(l <= 1);
  std::vector<std::vector<double>> energies;
  for(int N: N_list)
  {
    std::cout << "Calculating " << N << " Spins for lambda " << l << "." << std::endl;
    std::vector<double> energy;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es.compute(H0plusVaffplusVtf(N, s, l, p));
      double ev = es.eigenvalues()(0);
      energy.push_back(ev/N);
    }
    energies.push_back(energy);
  }
  std::cout << "Done." << std::endl;
  return energies;
}

std::vector<std::vector<double>> lambdaNotOneConcurrence(const double& l, const int& p, const std::vector<double>& s_list,  const std::vector<int>& N_list)
{
  // Calculates the rescaled concurrence for lambda=l
  assert(l >= 0);
  assert(l <= 1);
  std::vector<std::vector<BigDouble> > pascal, temp;
  std::vector<std::vector<double> > concurrences;
  for(int N: N_list)
  {
    temp = pascalTriangle(pascal.size(), N);
    pascal.insert(pascal.end(), temp.begin(), temp.end());
    std::cout << "Calculating concurrence " << N << " Spins." << std::endl;
    // std::vector<double> altconcurrences;
    std::vector<double> concurrence;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es.compute(H0plusVaffplusVtf(N, s, l, p));
      concurrence.push_back((N-1)*(calculateConcurrence(es, pascal, N)));

    }
    concurrences.push_back(concurrence);
  }
  std::cout << "Done." << std::endl;
  return concurrences;
}

// Author Johann Fredrik Jadebeck
//
// C++ port of quantum annealing python script.

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

#include "/home/fredrik/repos/gnuplot-cpp/gnuplot_i.hpp"
#include "quantumMechanics.hpp"

using namespace Eigen;

template<typename T>
std::vector<double> linspace(const T& s, const T& e, const int& n)
{
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

std::vector<std::vector<double>> create_const(const int& x, const int& y, const double& c)
{
  std::vector<std::vector<double>> one(x);
  for(auto &k : one)
  {
    k = std::vector<double>(y, c);
  }
  return one;
}

std::set<std::string> zeeman(const int& N, const double& m)
{
  // TODO add norm to every string, so we get unit vectors
  double S = (double)N/2.;
  std::string s="";
  // Figures out how many spins should be down.
  int n = S-m;
  std::set<std::string> permutations;
  for(int i=0; i<N-n; ++i)
  {
    s += "u";
  }
  for(int i=0; i<n; ++i)
  {
    s += "d";
  }
  // Finds permutations of s but if the string is not sorted,
  // the next paragraph will not find all permutations.
  permutations.insert(s);
  std::sort(s.begin(), s.end());
  do
  {
    permutations.insert(s);
  }
  while(std::next_permutation(s.begin(), s.end()));

  return permutations;
}

int cardaniac(const double& A, const double& B, const double& C, const double& D)
{
  return 0;
}

Matrix<double, Dynamic, Dynamic> H0(const int& N, const int& p)
{
  assert(N > 0);
  double S = double(N)/2.;
  Matrix<double, Dynamic, Dynamic> H0 = Sz(N);
  H0 /= S;
  MatrixPower<Matrix<double, Dynamic, Dynamic> > Apow(H0);
  H0 = Apow(p);
  H0 *= -N;
  return H0;
}

Matrix<double, Dynamic, Dynamic> Vtf(const int& N)
{
  assert(N > 0);
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> Vtf = Sx(N);
  Vtf /= S;
  Vtf *= -N;
  return Vtf;
}

Matrix<double, Dynamic, Dynamic> Vaff(const int& N)
{
  assert(N > 0);
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> Vaff = Sx(N);
  Vaff /= S;
  MatrixPower<Matrix<double, Dynamic, Dynamic> > Apow(Vaff);
  Vaff = Apow(2);
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

SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic>> diagonalize(Matrix<double, Dynamic, Dynamic> H)
{
  // BE VERY CAREFUL, THIS ONLY WORKS FOR SELF ADJOINT MATRICES,
  // WHICH THE HAMILTONIAN IS OF COUSE.
  SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic>> es;
  es.compute(H);
  return es;
}

Matrix<double, Dynamic, Dynamic> ket2dm(SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es, const int& N)
{
  return es.eigenvectors().col(0)*es.eigenvectors().col(0).transpose();
}

// Matrix<double, Dynamic, Dynamic> extract2qubitDm(Matrix<double, Dynamic, Dynamic> rho,
// 						 std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis,
// 						 const int& N)
// {
//   if(N==2)
//   {
//     // Nothing has to be done for N=2
//   }
//   else
//   {
//     //TODO implement partial trace
//     dm partial_rho(rho, zeemanBasis, N);
//     for(int i=0; i<N-2; ++i)
//     {
//       partial_rho = partial_rho.ptrace(0);
//     }
//     rho = partial_rho.nparray();
//   }
//   return rho;
// }

Matrix<double, Dynamic, Dynamic> extract2qubitDm(Matrix<double, Dynamic, Dynamic> rho,
						 std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis,
						 const int& N)
{
  if(N==2)
  {
    // Nothing has to be done for N=2
  }
  else
  {
    // for(int i=0; i<N-2; ++i)
    // {
    //   rho = ptrace(rho, 0);
    // }
    // //rho = partial_rho.nparray();
  }
  return rho;
}


double concurrence(Matrix<double, Dynamic, Dynamic> rho)
{
  Matrix<double, Dynamic, Dynamic> copy; // = rho;
  // Copies rho
  copy.resize(3, 3);
  copy.setZero();
  copy(0, 0) = rho(0, 0);
  copy(1, 0) = rho(1, 0);
  copy(2, 0) = rho(2, 0);
  copy(0, 1) = rho(0, 1);
  copy(1, 1) = rho(1, 1);
  copy(2, 1) = rho(2, 1);
  copy(0, 2) = rho(0, 2);
  copy(1, 2) = rho(1, 2);
  copy(2, 2) = rho(2, 2);

  // Applies the sigmay_i tensor sigmay_j to rho 
  rho.resize(4, 4);
  rho.setZero();
  rho(0, 0) = -copy(0, 2);
  rho(1, 0) = -copy(1, 2);
  rho(2, 0) = -copy(2, 2);
  rho(0, 1) = copy(0, 1);
  rho(1, 1) = copy(1, 1);
  rho(2, 1) = copy(2, 1);
  rho(0, 2) = -copy(0, 0);
  rho(1, 2) = -copy(1, 0);
  rho(2, 2) = -copy(2, 0);
  MatrixPower<Matrix<double, Dynamic, Dynamic> > Apow(rho);
  rho = Apow(2);
  EigenSolver<Matrix<double, Dynamic, Dynamic> > es;
  es.compute(rho);
  auto ev = es.eigenvalues();
  // Takes the sqrt and sorts the ev since they don't seem
  // to come out in order every time. Additionally
  // abs is applied since sometimes numbers close to 0
  // will be negative, which they of course can't be.
  std::vector<double> lambdas = {std::sqrt(std::abs(ev(0))),
				 std::sqrt(std::abs(ev(1))),
				 std::sqrt(std::abs(ev(2))),
				 std::sqrt(std::abs(ev(3)))} ;
  std::sort(lambdas.begin(), lambdas.end());
  double c = std::max(0., lambdas[3] - lambdas[2] - lambdas[1] - lambdas[0]);
  return c;
}

double calculateConcurrence(SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es,
			    std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis,
			    const int& N)
{
  Matrix<double, Dynamic, Dynamic> rho = ket2dm(es, N);
  rho = extract2qubitDm(rho, zeemanBasis, N);
  double trace = rho.trace();
  assert(std::abs(trace-1.) < 0.0001);
  double c = concurrence(rho);
  return c;
}

void lambdaOne(const int& p)
{
  // Calculates the groundstate energy for lambda=1
  std::vector<double> s_list = linspace(0, 1, 101);
  std::vector<int> N_list = {2, 4, 8, 16, 32, 64, 128, 256};
  std::vector<std::vector<double>> energies;
  for(int N: N_list)
  {
    std::cout << "Calculating " << N << " Spins." << std::endl;
    std::vector<double> energy;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es = diagonalize(H0plusVtf(N, s, p));
      double ev = es.eigenvalues()(0);
      energy.push_back(ev/N);
    }
    energies.push_back(energy);
  }
  std::cout << "Done." << std::endl;
  Gnuplot gp("Energy per spin");
  std::ostringstream s;
  s << "Energy per Spin ";
  auto title = s.str();
  gp.set_title(title);
  gp.set_xlabel("s");
  for(int i=0; i<N_list.size(); ++i)
  {
    std::ostringstream s2;
    s2 << N_list[i] << " Spins";
    gp.set_style("lines").plot_xy(s_list, energies[i], s2.str());
  }
  gp.unset_smooth();
  gp.showonscreen();
  std::cout << "Press Enter to exit." << std::endl;
  // "read" for max linux, "pause" for windows
  std::system("read");
}

void lambdaOneConcurrence(const int& p)
{
  // Calculates the rescaled concurrence for lambda=1
  std::vector<double> s_list = linspace(0, 1, 101);
  std::vector<int> N_list = {2};
  std::vector<std::vector<double> > concurrences;
  std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis;
  for(int N: N_list)
  {
    // TODO calculate zeeman basis once here up until N
    std::cout << "Calculating concurrence" << N << " Spins." << std::endl;
    std::vector<double> concurrence;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es = diagonalize(H0plusVtf(N, s, p));
      concurrence.push_back(calculateConcurrence(es, zeemanBasis, N)*(N-1));
    }
    concurrences.push_back(concurrence);
  }
  std::cout << "Done." << std::endl;
  Gnuplot gp("Rescaled concurrence Cr");
  std::ostringstream s;
  s << "Energy per Spin ";
  auto title = s.str();
  gp.set_title(title);
  gp.set_xlabel("s");
  gp.set_ylabel("Cr");
  for(int i=0; i<N_list.size(); ++i)
  {
    std::ostringstream s2;
    s2 << N_list[i] << " Spins";
    gp.set_style("lines").plot_xy(s_list, concurrences[i], s2.str());
  }
  gp.unset_smooth();
  gp.showonscreen();
  std::cout << "Press Enter to exit." << std::endl;
  // "read" for max linux, "pause" for windows
  std::system("read");
}

void lambdaNotOne(const int& p)
{
  // Calculates the groundstate energy for lambda!=1
  std::vector<double> s_list = linspace(0, 1, 101);
  std::vector<double> l_list = linspace(0, 1, 6);
  std::vector<int> N_list = {2, 4, 8, 16, 32, 64, 128};
  for(double l : l_list)
  {
    std::vector<std::vector<double>> energies;
    for(int N: N_list)
    {
      std::cout << "Calculating " << N << " Spins for lambda "
		<< l << "." << std::endl;
      std::vector<double> energy;
      for(double s: s_list)
      {
	SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
	es = diagonalize(H0plusVaffplusVtf(N, s, l, p));
	double ev = es.eigenvalues()(0);
	energy.push_back(ev/N);
      }
      energies.push_back(energy);
    }
    std::cout << "Done." << std::endl;
    Gnuplot gp("Energy per spin");
    std::ostringstream s;
    s << "Energy per Spin ";
    auto title = s.str();
    gp.set_title(title);
    gp.set_xlabel("s");
    for(int i=0; i<N_list.size(); ++i)
    {
      std::ostringstream s2;
      s2 << N_list[i] << " Spins";
      gp.set_style("lines").plot_xy(s_list, energies[i], s2.str());
    }
    gp.unset_smooth();
    gp.showonscreen();
    std::cout << "Press Enter to exit." << std::endl;
    // "read" for max linux, "pause" for windows
    std::system("read");
  }
}

void lambdaNotOneConcurrence(const int& p)
{
  // Calculates the rescaled concurrence for lambda!=1
  std::vector<double> s_list = linspace(0, 1, 101);
  std::vector<double> l_list = linspace(0, 1, 6);
  std::vector<int> N_list = {2};
  std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis;
  for(double l : l_list)
  {
    std::vector<std::vector<double> > concurrences;
    for(int N: N_list)
    {
      std::cout << "Calculating concurrence" << N << " Spins." << std::endl;
      std::vector<double> concurrence;
      for(double s: s_list)
      {
	SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
	es = diagonalize(H0plusVaffplusVtf(N, s, l, p));
	concurrence.push_back(calculateConcurrence(es, zeemanBasis, N)*(N-1));
      }
      concurrences.push_back(concurrence);
    }
    std::cout << "Done." << std::endl;
    Gnuplot gp("Rescaled concurrence Cr");
    std::ostringstream s;
    s << "Rescaled Concurrece for lambda " << l << ".";
    auto title = s.str();
    gp.set_title(title);
    gp.set_xlabel("s");
    gp.set_ylabel("Cr");
    for(int i=0; i<N_list.size(); ++i)
    {
      std::ostringstream s2;
      s2 << N_list[i] << " Spins";
      gp.set_style("points").plot_xy(s_list, concurrences[i], s2.str());
    }
    gp.unset_smooth();
    gp.showonscreen();
    std::cout << "Press Enter to exit." << std::endl;
    // "read" for max linux, "pause" for windows
    std::system("read");
  }
}

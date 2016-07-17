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
  SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic>> es;
  es.compute(H);
  return es;
}

void lambdaOne(const int& p)
{
  //Calculates the groundstate energy for lambda=1
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
    gp.set_style("points").plot_xy(s_list, energies[i], s2.str());
  }
  gp.unset_smooth();
  gp.showonscreen();
  std::cout << "Press Enter to exit." << std::endl;
  double e;
  std::cin >> e;
}

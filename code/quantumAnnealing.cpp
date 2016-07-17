// Author Johann Fredrik Jadebeck
//
// C++ port of quantum annealing python script.

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <Eigen/Core>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <iostream>
#include <set>
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
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> H0 = Sz(N).pow(p);
  H0 *= -N/S;
  return H0;
}

int main()
{
}

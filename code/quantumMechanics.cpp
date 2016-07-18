// Author Fredrik Jadebeck
//
// Source for the quantumMechanics.py port

#include <assert.h>
#include <cmath>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <vector>

#include "quantumMechanics.hpp"

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
    Sx(i, i+1) = mSxn(m[i], m[i+1], S);
    Sx(i+1, i) = mSxn(m[i+1], m[i], S);
  }
  return Sx;
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


Matrix<double, Dynamic, Dynamic> ptrace(const Matrix<double, Dynamic, Dynamic>& rho, std::vector<std::vector<int> > pascal, const int& N)
{
  //N is spins+1
  Matrix<double, Dynamic, Dynamic> res;
  res.resize(N-1, N-1);
  res.setZero();
  std::vector<int> New = pascal[N-2];
  std::vector<int> Old = pascal[N-1];
  double temp = 0;
  // TODO use symmetry to save calculations after its working
  for(int i=0; i<N; ++i)
  {
    for(int j=0; j<N; ++j)
    {
      double factor = std::sqrt(Old[i]);
      factor*= std::sqrt(Old[j]);
      if((i < N-1) and (j < N-1))
      {	
      	temp = rho(i, j) / factor;
      	temp *= std::sqrt(New[i]);
      	temp *= std::sqrt(New[j]);
	res(i, j) += temp;
      }
      if((i > 0) and (j > 0))
      {
      	temp = rho(i, j) / factor;
      	temp *= std::sqrt(New[i-1]);
      	temp *= std::sqrt(New[j-1]);
	res(i-1, j-1) += temp;
      }
    }
  }
  return res;
}

std::vector<std::vector<int> > pascalTriangle(const int& N)
{
  // Calculates pascal triangle up to N spins which means N+1 lines
  // in O(N^2). Be careful of integerowerflow though
  std::vector<std::vector<int> > triangle;
  // Starts at line 0 even if its not necessary because
  // this leads to the vector index being equal to the
  // number of spins.
  for(int line=0; line<N+1; line++)
    {
      int C = 1;
      std::vector<int> lin;
      for(int i=1; i<line+2; i++)
    {
      lin.push_back(C);
      C = C*(line -i + 1)/i;
    }
    triangle.push_back(lin);
  }

  return triangle;
}

std::vector<std::vector<int> > pascalTriangle(const int& prev, const int& N)
{
  // Calculates pascal triangle up to N spins starting from line prev which means N+1 lines
  // in O(N^2). Be careful of integerowerflow though
  std::vector<std::vector<int> > triangle;
  // Starts at line 0 even if its not necessary because
  // this leads to the vector index being equal to the
  // number of spins.
  for(int line=prev; line<N+1; line++)
    {
      int C = 1;
      std::vector<int> lin;
      for(int i=1; i<line+2; i++)
    {
      lin.push_back(C);
      C = C*(line -i + 1)/i;
    }
    triangle.push_back(lin);
  }

  return triangle;
}

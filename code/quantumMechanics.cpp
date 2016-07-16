// Author Fredrik Jadebeck
//
// Headerfile for the quantumMechanics.py port
#ifndef QUANTUMMECHANICS_H
#define QUANTUMMECHANICS_H

#include <assert.h>
#include <Eigen/Core>
#include <set>
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

}

double mSzn(const double& m, const double& n, const double& S)
{

}

// auto Sx(const int& N)
// {

// }

// auto Sz(const int& N)
// {

// }
#endif

// Author Fredrik Jadebeck
//
// Unit tests for the c++ port of quantumMechanics
#include "quantumMechanics.hpp"
#include "quantumAnnealing.hpp"
#define BOOST_TEST_MODULE quantumMechanicsTest
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE GRNN test suite 

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <Eigen/Core>
#include <vector>

BOOST_AUTO_TEST_CASE(test_H0)
{
  std::vector<int> N_list = {2, 3, 64};
  for(int N : N_list)
  {
    int p = 5;
    double S = double(N)/2;
    std::vector<double> m = gen_m(N);
    Matrix<double, Dynamic, Dynamic> expected;
    expected.resize(N+1, N+1);
    expected.setZero();
    for(int i=0; i<N+1; ++i)
    {
      expected(i, i) = -N*std::pow(((double)m[i]/S), p);
    }
    Matrix<double, Dynamic, Dynamic> actual = H0(N, p);
    BOOST_CHECK(expected.isApprox(actual, 0.00001));
  }
}

BOOST_AUTO_TEST_CASE(test_Vtf)
{
  std::vector<int> N_list = {2, 3, 64};
  for(int N : N_list)
  {
    double S = double(N)/2;
    std::vector<double> m = gen_m(N);
    Matrix<double, Dynamic, Dynamic> expected;
    expected.resize(N+1, N+1);
    expected.setZero();
    for(int i=0; i<N; ++i)
    {
      expected(i, i+1) = -N*mSxn(m[i], m[i+1], S)/S;
      expected(i+1, i) = -N*mSxn(m[i+1], m[i], S)/S;
    }
    Matrix<double, Dynamic, Dynamic> actual = Vtf(N);
    BOOST_CHECK(expected.isApprox(actual, 0.00001));
  }
}

BOOST_AUTO_TEST_CASE(test_Vaff)
{
  std::vector<int> N_list = {2, 3, 64};
  for(int N : N_list)
  {
    double S = double(N)/2;
    std::vector<double> m = gen_m(N);
    Matrix<double, Dynamic, Dynamic> expected;
    expected.resize(N+1, N+1);
    expected.setZero();
    for(int i=0; i<N; ++i)
    {
      expected(i, i+1) = mSxn(m[i], m[i+1], S)/S;
      expected(i+1, i) = mSxn(m[i+1], m[i], S)/S;
    }
    expected *= expected;
    expected *= N;
    Matrix<double, Dynamic, Dynamic> actual = Vaff(N);
    BOOST_CHECK(expected.isApprox(actual, 0.00001));
  }
}

BOOST_AUTO_TEST_CASE(test_concurrence)
{
  double expected;
  double actual;
  Matrix<double, Dynamic, Dynamic> rho;
  rho.resize(3, 3);
  
  expected = 0;
  rho.setZero();
  actual = concurrence(rho);
  assert(actual == expected);

  expected = 1;
  rho.setZero();
  rho(0, 2) = 1;
  actual = concurrence(rho);
  assert(actual == expected);

  expected = 0.4471470773590325;
  rho.setZero();
  rho(0, 0) = 0.947195;
  rho(0, 1) = -0.00638849;
  rho(0, 2) = -0.223552;
  rho(1, 0) = -0.00638849;
  rho(1, 1) = 4.3099e-05;
  rho(1, 2) = 0.00150778;
  rho(2, 0) = -0.223552;
  rho(2, 1) = 0.00150778;
  rho(2, 2) = 0.0527616;
}

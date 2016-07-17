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


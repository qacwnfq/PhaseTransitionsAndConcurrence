// Author Fredrik Jadebeck
//
// Unit tests for the c++ port of quantumMechanics

#include "quantumAnnealing.hpp"

#define BOOST_TEST_MODULE quantumMechanicsTest
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE GRNN test suite 

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <vector>

using namespace Eigen;


BOOST_AUTO_TEST_CASE(test_gen_m)
{
  // tests N=2 case
  std::vector<double> actual = gen_m(2);
  std::vector<double> expected;
  expected.push_back(-1);
  expected.push_back(0);
  expected.push_back(1);
  BOOST_CHECK(actual == expected);

  // tests N=3 case
  actual = gen_m(3);
  expected.clear();
  expected.push_back(-1.5);
  expected.push_back(-0.5);
  expected.push_back(0.5);
  expected.push_back(1.5);
  BOOST_CHECK(actual == expected);
}

BOOST_AUTO_TEST_CASE(test_kronecker)
{
  // tests int cases
  int actual = kronecker(1, 0);
  int expected = 0;
  BOOST_CHECK(actual == expected);
  
  actual = kronecker(1, 1);
  expected = 1;
  BOOST_CHECK(actual == expected);

  // tests double cases
  actual = kronecker(1.5, 0.5);
  expected = 0;
  BOOST_CHECK(actual == expected);

  actual = kronecker(-1.5, -1.5);
  expected = 1;
  BOOST_CHECK(actual == expected);
}

BOOST_AUTO_TEST_CASE(test_mSxn)
{
  BOOST_CHECK(0 == mSxn(0, 3, 6));
  BOOST_CHECK(0.5*std::sqrt(4*4+4-2) == mSxn(1, 2, 4));
  BOOST_CHECK(0.5*std::sqrt(10*10+10-2) == mSxn(1, 2, 10));
  BOOST_CHECK(0.5*std::sqrt(100*100+100-99*100) == mSxn(99, 100, 100));
  BOOST_CHECK(1 == mSxn(2, 1, 2));
  BOOST_CHECK(1 == mSxn(-2, -1, 2));
  BOOST_CHECK(0.5*std::sqrt(2) == mSxn(0, 1, 1));
}

BOOST_AUTO_TEST_CASE(test_mSzn)
{
  BOOST_CHECK(0 == mSzn(0, 1, 5));
  BOOST_CHECK(0 == mSzn(99, 6, 100));
  BOOST_CHECK(100 == mSzn(100, 100, 100));
  BOOST_CHECK(100 == mSzn(100, 100, 1090));
  BOOST_CHECK(3 == mSzn(3, 3, 10));
  BOOST_CHECK(-10 == mSzn(-10, -10, 100));
  BOOST_CHECK(0 == mSzn(0, 0, 10000));
}

BOOST_AUTO_TEST_CASE(test_Sx)
{
  int N = 2;
  Matrix<double, 3, 3> expected;
  expected.setZero();
  expected(0, 1) = 0.5*std::sqrt(2);
  expected(1, 0) = 0.5*std::sqrt(2);
  expected(1, 2) = 0.5*std::sqrt(2);
  expected(2, 1) = 0.5*std::sqrt(2);
  Matrix<double, Dynamic, Dynamic> actual = Sx(N);
  BOOST_CHECK(expected == actual);

  N = 3;
  Matrix<double, 4, 4> expected2;
  expected2.setZero();
  expected2(0, 1) = std::sqrt(3)/2;
  expected2(1, 0) = std::sqrt(3)/2;
  expected2(1, 2) = 1;
  expected2(2, 1) = 1;
  expected2(2, 3) = std::sqrt(3)/2;
  expected2(3, 2) = std::sqrt(3)/2;
  actual = Sx(N);
  BOOST_CHECK(expected2 == actual);
}

BOOST_AUTO_TEST_CASE(test_Sy)
{
  const std::complex<double> I = {0.0, 1.0};
  int N = 2;
  Matrix<std::complex<double>, 3, 3> expected;
  expected.setZero();
  expected(0, 1) = 0.5*std::sqrt(2)/I;
  expected(1, 0) = -0.5*std::sqrt(2)/I;
  expected(1, 2) = 0.5*std::sqrt(2)/I;
  expected(2, 1) = -0.5*std::sqrt(2)/I;
  Matrix<std::complex<double>, Dynamic, Dynamic> actual = Sy(N);
  BOOST_CHECK(expected == actual);

  N = 3;
  Matrix<std::complex<double>, 4, 4> expected2;
  expected2.setZero();
  expected2(0, 1) = std::sqrt(3)/(2.*I);
  expected2(1, 0) = -std::sqrt(3)/(2.*I);
  expected2(1, 2) = 1./I;
  expected2(2, 1) = -1./I;
  expected2(2, 3) = std::sqrt(3)/(2.*I);
  expected2(3, 2) = -std::sqrt(3)/(2.*I);
  actual = Sy(N);
  BOOST_CHECK(expected2 == actual);
}


BOOST_AUTO_TEST_CASE(test_Sz)
{
  int N = 2;
  Matrix<double, 3, 3> expected;
  expected.setZero();
  expected(0, 0) = -1;
  expected(1, 1) = 0;
  expected(2, 2) = 1;
  Matrix<double, Dynamic, Dynamic> actual = Sz(N);
  BOOST_CHECK(expected == actual);

  N = 3;
  Matrix<double, 4, 4> expected2;
  expected2.setZero();
  expected2(0, 0) = -1.5;
  expected2(1, 1) = -0.5;
  expected2(2, 2) = 0.5;
  expected2(3, 3) = 1.5;
  actual = Sz(N);
  BOOST_CHECK(expected2 == actual);
}

BOOST_AUTO_TEST_CASE(test_expectationValue)
{
  Matrix<double, Dynamic, Dynamic> state;
  Matrix<complex, Dynamic, Dynamic> op;
  state.resize(3, 1);
  op.resize(3, 3);
  complex expected;
  complex actual;

  state.setZero();
  op.setZero();
  state(0, 0) = 1;
  op(0, 1) = 1/std::sqrt(2);
  op(1, 0) = 1/std::sqrt(2);
  op(1, 2) = 1/std::sqrt(2);
  op(2, 1) = 1/std::sqrt(2);
  expected = 0.;
  actual = expectationValue(state, op);
  BOOST_CHECK(expected == actual);

  state(0, 0) = 1;
  state(1, 0) = 1;
  state(2, 0) = 1;
  expected = 4./std::sqrt(2);
  actual = expectationValue(state, op);
  BOOST_CHECK(actual == expected);
}

BOOST_AUTO_TEST_CASE(test_twoSystemRho)
{
  Matrix<double, Dynamic, Dynamic> H = -H0(2, 5);
  Matrix<double, Dynamic, Dynamic> e;
  e.resize(4, 4);
  e.setZero();
  e(3, 3) = 1;
  auto expected = dM2c(e);
  SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
  es.compute(H);
  auto vector = es.eigenvectors().col(0);
  auto actual = twoSystemRho(vector);
  BOOST_CHECK(expected.isApprox(actual, 0.00001));
}

BOOST_AUTO_TEST_CASE(test_ptrace)
{
  int N = 2;
  Matrix3d rho;
  rho << 1, 2, 3, 4, 5, 6, 7, 8, 9;
  Matrix2d expected;
  expected(0, 0) = rho(0, 0) + rho(1, 1)/2;
  expected(0, 1) = 1/std::sqrt(2)*(rho(0, 1)+rho(1, 2));
  expected(1, 0) = 1/std::sqrt(2)*(rho(1, 0)+rho(2, 1));
  expected(1, 1) = rho(2, 2) + rho(1, 1)/2;
  std::vector<std::vector<long double> > pascal = pascalTriangle(0, N);
  Matrix<double, Dynamic, Dynamic> actual;
  actual = ptrace(rho, pascal, N+1);
  BOOST_CHECK(expected.isApprox(actual, 0.0001));
}

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

BOOST_AUTO_TEST_CASE(test_concurrence_alternative)
{
  std::vector<double> s_list = linspace(0, 1, 11);
  std::vector<double> l_list = linspace(0, 1, 6);
  std::vector<int> N_list = {2, 4, 8, 16};
  int p = 7;
  for(int l: l_list)
  {
    std::vector<std::vector<BigDouble> > pascal, temp;
    std::vector<std::vector<double> > concurrences;
    std::vector<std::vector<double> > altConcurrences;
    for(int N: N_list)
    {
      temp = pascalTriangle(pascal.size(), N);
      pascal.insert(pascal.end(), temp.begin(), temp.end());
      std::vector<double> altconcurrences;
      std::vector<double> concurrence;
      for(double s: s_list)
      {
	SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
	es.compute(H0plusVaffplusVtf(N, s, l, p));
	concurrence.push_back((N-1)*(calculateConcurrence(es, pascal, N)));
	// Alternative way of calculating concurrence
	altconcurrences.push_back(altConcurrence(es.eigenvectors().col(0))*(N-1));
	// Checks if both ways agree
	assert(std::abs(concurrence.back() - altconcurrences.back()) < 0.0001);
	assert(concurrence.back()/(N-1) <= 1.);
      }
    }
  }
}

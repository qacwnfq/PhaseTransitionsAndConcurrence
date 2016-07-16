// Author Fredrik Jadebeck
//
// Unit tests for the c++ port of quantumMechanics
#include "quantumMechanics.hpp"
#define BOOST_TEST_MODULE quantumMechanicsTest
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE GRNN test suite 

#include <boost/test/unit_test.hpp>
#include <cmath>
#include <vector>

//TODO remove this later
#include <iostream>

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

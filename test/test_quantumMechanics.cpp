// Author Fredrik Jadebeck
//
// Unit tests for the c++ port of quantumMechanics
#include "quantumMechanics.hpp"
#define BOOST_TEST_MODULE quantumMechanicsTest
#define BOOST_TEST_DYN_LINK 
#define BOOST_TEST_MODULE GRNN test suite 

#include <boost/test/unit_test.hpp>
#include <vector>

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

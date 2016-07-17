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
  std::vector<int> N_list = {4};
  for(int N : N_list)
  {
    int p = 5;
    double S = double(N)/2;
    std::vector<double> m = gen_m(N);
    Matrix<double, Dynamic, Dynamic> expected;
    expected.setZero();
    expected.resize(N+1, N+1);
    for(int i=0; i<N+1; ++i)
    {
      expected(i, i) = -N*std::pow(((double)m[i]/S), p);
    }
    Matrix<double, Dynamic, Dynamic> actual = H0(N, p);
    std::cout << actual << std::endl << std::endl;
    std::cout << expected << std::endl << std::endl;
    std::cout << actual-expected << std::endl;
    std::cout << expected.isApprox(actual, 0.001) << std::endl;
    BOOST_CHECK(expected.isApprox(actual, 0.001));
  }
}

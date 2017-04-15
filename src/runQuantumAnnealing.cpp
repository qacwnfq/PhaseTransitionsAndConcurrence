#include <iostream>
#include <fstream>
#include <string>
#include <cstdlib>

#include "quantumAnnealing.hpp"

int main(int argc, char* argv[])
{
  int p = 5;
  // This program accepts one parameter for p
  if(argc == 2)
  {
    p = std::atoi(argv[1]);
  }
  std::cout << "Running for p=" << std::to_string(p) << std::endl;

  // Usage example:
  std::vector<double> s_list = linspace(0, 1, 501);
  std::vector<double> l_list = linspace(0, 1, 6);
  std::vector<int> N_list = {2, 4, 8, 16, 32, 64, 128, 256};
  std::vector<std::vector<double>> energies = lambdaOne(p, s_list, N_list);
  std::vector<std::vector<double>> concurrences = lambdaOneConcurrence(p, s_list, N_list);
  for(double l : l_list)
  {
    std::vector<std::vector<double>> energiesLambdaNotOne = lambdaNotOne(l, p, s_list, N_list);
    if(l==0.)
    {
      //No concurrence if l=0
      continue;
    }
    else
    {
      std::vector<std::vector<double>> concurrencesLambdaNotOne = lambdaNotOneConcurrence(l, p, s_list, N_list);
    }
  }
}

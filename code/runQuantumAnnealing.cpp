#include <iostream>
#include <string>
#include <cstdlib>

#include "quantumAnnealing.hpp"

int main(int argc, char* argv[])
{
  int p = 5;
  if(argc == 2)
  {
    p = std::atoi(argv[1]);
  }
  // runs groundstate energy calculation for p=5
  //　lambdaOne(5);
  // lambdaOneConcurrence(5);
  // lambdaNotOne(5);
  std::cout << "Running for p=" << std::to_string(p) << std::endl;
  lambdaNotOneConcurrence(p);
}

#include <iostream>
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
  // lambdaOneConcurrence(p);
  lambdaNotOneConcurrence(p);
}

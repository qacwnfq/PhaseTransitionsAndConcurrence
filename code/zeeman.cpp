// Author Fredrik Jadebeck
//
// This is a port of the zeeman function to C++

#include <algorithm>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <set>
#include <string>

std::set<std::string> zeeman(const int& N, const double& m)
{
  double S = (double)N/2.;
  std::cout << S << std::endl;
  std::cout << m << std::endl;
  std::string s="";
  // Figures out how many spins should be down.
  int n = S-m;
  std::set<std::string> permutations;
  for(int i=0; i<N-n; ++i)
  {
    s += "u";
  }
  for(int i=0; i<n; ++i)
  {
    s += "d";
  }
  // Finds permutations of s but if the string is not sorted,
  // the next paragraph will not find all permutations.
  permutations.insert(s);
  std::sort(s.begin(), s.end());
  do
  {
    permutations.insert(s);
  }
  while(std::next_permutation(s.begin(), s.end()));

  return permutations;
}

// port of gen_m to c++
std::set<double> gen_m(int N)
{
  assert(N > 0);
  N = (double)N;
  double S = (double)N/2.;
  std::set<double> m;
  for(int i=0; i<=N; ++i)
  {
    m.insert(-S+i);
  }
  return m;
}


void run()
{
  std::ofstream myfile;
  myfile.open("data/zeemanbasis.csv");
  int maxN = 6;
  for(int N=5; N<maxN; ++N)
  {
    std::set<double> s = gen_m(N);
    std::set<double>::iterator it;
    myfile << N << ",";
    for(it=s.begin();it!=s.end(); ++it)
    {
      std::set<std::string> zeemanBasisVectors;
      zeemanBasisVectors = zeeman(N, *it);
      std::set<std::string>::iterator zit;
      for(zit=zeemanBasisVectors.begin(); zit!=zeemanBasisVectors.end(); zit++)
      {
	std::cout << *zit << std::endl;
	myfile << *zit << " ";
      }
      myfile << ",";
    }
    for(int i=N+1; i<=maxN;++i)
      myfile << ",";
    myfile << std::endl;
  }
    myfile.close();
}

int main()
{
  run();
}

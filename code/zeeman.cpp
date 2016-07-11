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
  int maxN = 12;
  // Writes column header row. There are N+1 columns, same
  // as the number of values for m.
  for(int i=0; i<=maxN; ++i)
  {
    myfile << i << ',';
  }
  myfile << std::endl;
  
  for(int N=2; N<maxN; ++N)
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
	myfile << *zit << " ";
      }
      myfile << ",";
    }
    // The loop adds more columns so that python pandas dataframe
    // knows the max number of columns is achieved for every row.
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

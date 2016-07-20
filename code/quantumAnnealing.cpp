// Author Johann Fredrik Jadebeck
//
// C++ port of quantum annealing python script.

#include <algorithm>
#include <assert.h>
#include <cmath>
#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MatrixFunctions>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "/home/fredrik/repos/gnuplot-cpp/gnuplot_i.hpp"
#include "quantumMechanics.hpp"

using namespace Eigen;

// TODO add typedef

template<typename T>
std::vector<double> linspace(const T& s, const T& e, const int& n)
{
  // This is a useful template to recreate numpys linspace from python
  double start = static_cast<double>(s);
  double end = static_cast<double>(e);
  double num = static_cast<double>(n);
  double delta = (end - start)/(num-1);

  std::vector<double> linspaced(num);
  for(int i=0; i<num; ++i)
  {
    linspaced[i] = start + delta*i;
  }
  linspaced[end];
  return linspaced;
}


std::vector<double> readLimit(std::string title)
{
  // Reads the limit for concurrence
  // which the python script quantumAnnealing.py
  // calculates into a vector.
  std::ifstream file;
  file.open(title);
  std::vector<double> result;
  std::string line;

  double c;
  // Skips header line
  getline(file, line);
  while(std::getline(file, line))
  {
    // Reads line
    std::stringstream  lineStream(line);
    std::string        cell;
    while(std::getline(lineStream, cell, ','))
    {
      // Reads every cell in a line.
      // The concurrence values are
      // stored in the last cell.
      c = std::stod(cell);
    }
    result.push_back(c);
  }
  file.close();
  return result;
}


Matrix<double, Dynamic, Dynamic> H0(const int& N, const int& p)
{
  // Returns matrix representation of the target Hamiltonian H0
  double S = double(N)/2.;
  Matrix<double, Dynamic, Dynamic> H0 = Sz(N);
  H0 /= S;
  MatrixPower<Matrix<double, Dynamic, Dynamic> > Apow(H0);
  H0 = Apow(p);
  H0 *= -N;
  return H0;
}

Matrix<double, Dynamic, Dynamic> Vtf(const int& N)
{
  // Returns matrix representation of the transverse field Vtf
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> Vtf = Sx(N);
  Vtf /= S;
  Vtf *= -N;
  return Vtf;
}

Matrix<double, Dynamic, Dynamic> Vaff(const int& N)
{
  // Returns matrix representation of the antiferromagnetic term Vaff
  double S = double(N)/2;
  Matrix<double, Dynamic, Dynamic> Vaff = Sx(N);
  Vaff /= S;
  MatrixPower<Matrix<double, Dynamic, Dynamic> > Apow(Vaff);
  Vaff = Apow(2);
  Vaff *= N;
  return Vaff;
}

Matrix<double, Dynamic, Dynamic> H0plusVtf(const int& N, const double& s, const int& p)
{
  return s*H0(N, p) + (1-s)*Vtf(N);
}

Matrix<double, Dynamic, Dynamic> H0plusVaffplusVtf(const int& N, const double& s, const double& l, const int& p)
{
  return s*l*H0(N, p) + s*(1-l)*Vaff(N) + (1-s)*Vtf(N);
}

SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic>> diagonalize(Matrix<double, Dynamic, Dynamic> H)
{
  // Uses the fact, that the hamiltonian is selfadjoint.
  SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic>> es;
  es.compute(H);
  return es;
}

Matrix<double, Dynamic, Dynamic> ket2dm(SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es, const int& N)
{
  // Takes a ket and creates a density matrix from it.
  return es.eigenvectors().col(0)*es.eigenvectors().col(0).transpose();
}

Matrix<double, Dynamic, Dynamic> extract2particleDm(Matrix<double, Dynamic, Dynamic> rho,
						 std::vector<std::vector<unsigned long long int> > pascal,
						 const int& N)
{
  // Takes a density matrix in zee man basis and
  // traces out all particles expect for 2.
  int SpinsPlusOne = N+1;
  for(int i=0; i<N-2; ++i)
  {
    // ptrace() expects spins+1 to know
    // the size of rho ((N+1)x(N+1))
    rho = ptrace(rho, pascal, SpinsPlusOne);
    SpinsPlusOne--;
  }
  return rho;
}


double concurrence(Matrix<double, Dynamic, Dynamic> rho)
{
  // Calculates concurrence for a 2 particle
  // density matrix in the zeeman basis.
  Matrix<double, Dynamic, Dynamic> copy = rho;
  rho.resize(4, 4);
  rho.setZero();
  // Applies the sigmay_i tensor sigmay_j to rho 
  rho(0, 0) = -copy(0, 2);
  rho(1, 0) = -copy(1, 2);
  rho(2, 0) = -copy(2, 2);
  rho(0, 1) = copy(0, 1);
  rho(1, 1) = copy(1, 1);
  rho(2, 1) = copy(2, 1);
  rho(0, 2) = -copy(0, 0);
  rho(1, 2) = -copy(1, 0);
  rho(2, 2) = -copy(2, 0);

  // Calculates R matrix and store it in rho variable.
  MatrixPower<Matrix<double, Dynamic, Dynamic> > Apow(rho);
  rho = Apow(2);
  // Uses the regular eigensolver here,
  // because the R matrix is not selfadjoint.
  EigenSolver<Matrix<double, Dynamic, Dynamic> > es;
  es.compute(rho);
  auto ev = es.eigenvalues();
  // Applies std::abs() before taking the sqrt to avoid
  // complex numbers. When the eigenvalues are close to 0
  // the floating point precision will sometimes lead to
  // negative values, while they should actually be zero.
  std::vector<double> lambdas = {std::sqrt(std::abs(ev(0))),
				 std::sqrt(std::abs(ev(1))),
				 std::sqrt(std::abs(ev(2))),
				 std::sqrt(std::abs(ev(3)))};
  std::sort(lambdas.begin(), lambdas.end());
  double c = std::max(0., lambdas[3] - lambdas[2] - lambdas[1] - lambdas[0]);
  return c;
}

double calculateConcurrence(SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es,
			    std::vector<std::vector<unsigned long long int> > pascal,
			    const int& N)
{
  Matrix<double, Dynamic, Dynamic> rho = ket2dm(es, N);
  rho = extract2particleDm(rho, pascal, N);
  double trace = rho.trace();
  // This assert protects from integer overflow.
  assert(std::abs(trace-1.) < 0.0001);
  double c = concurrence(rho);
  return c;
}

void lambdaOne(const int& p)
{
  // Calculates the groundstate energy for lambda=1
  std::vector<double> s_list = linspace(0, 1, 501);
  std::vector<int> N_list = {2, 4, 8, 16, 32, 64, 128, 256};
  std::vector<std::vector<double>> energies;
  for(int N: N_list)
  {
    std::cout << "Calculating " << N << " Spins." << std::endl;
    std::vector<double> energy;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es = diagonalize(H0plusVtf(N, s, p));
      double ev = es.eigenvalues()(0);
      energy.push_back(ev/N);
    }
    energies.push_back(energy);
  }
  std::cout << "Done." << std::endl;
  Gnuplot gp("Energy per spin");
  std::ostringstream s;
  s << "Energy per Spin ";
  auto title = s.str();
  gp.set_title(title);
  gp.set_xlabel("s");
  for(int i=0; i<N_list.size(); ++i)
  {
    std::ostringstream s2;
    s2 << N_list[i] << " Spins";
    gp.set_style("lines").plot_xy(s_list, energies[i], s2.str());
  }
  gp.unset_smooth();
  gp.showonscreen();
  std::cout << "Press Enter to exit." << std::endl;
  // "read" for max linux, "pause" for windows
  std::system("read");
}

void lambdaOneConcurrence(const int& p)
{
  // Calculates the rescaled concurrence for lambda=1
  std::vector<double> s_list = linspace(0, 1, 501);
  std::vector<int> N_list = {2, 4, 8, 16, 32, 62};

  std::vector<std::vector<double> > concurrences;
  std::vector<std::vector<unsigned long long int> > pascal, temp;
  for(int N: N_list)
  {
    temp = pascalTriangle(pascal.size(), N);
    pascal.insert(pascal.end(), temp.begin(), temp.end());
    std::cout << "Calculating concurrence " << N << " Spins." << std::endl;
    std::vector<double> concurrence;
    for(double s: s_list)
    {
      SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
      es = diagonalize(H0plusVtf(N, s, p));
      concurrence.push_back(calculateConcurrence(es, pascal, N)*(N-1));
    }
    concurrences.push_back(concurrence);
  }
  std::cout << "Done." << std::endl;
  Gnuplot gp("Rescaled concurrence Cr");
  std::ostringstream s;
  s << "Rescaled concurrence for lambda=1";
  auto title = s.str();
  gp.set_title(title);
  gp.set_xlabel("s");
  gp.set_ylabel("Cr");
  for(int i=0; i<N_list.size(); ++i)
  {
    std::ostringstream s2;
    s2 << N_list[i] << " Spins";
    gp.set_style("lines").plot_xy(s_list, concurrences[i], s2.str());
  }

  title = ("../results/concurrence/p" + std::to_string(p) + "/lambda1limit.csv");
  std::cout << "reading " << title << std::endl;
  std::vector<double> limit = readLimit(title);

  gp.set_style("lines").plot_xy(s_list, limit, "limit");
  gp.unset_smooth();
  gp.showonscreen();
  std::cout << "Press Enter to exit." << std::endl;
  // "read" for max linux, "pause" for windows
  std::system("read");
}

void lambdaNotOne(const int& p)
{
  // Calculates the groundstate energy for lambda!=1
  std::vector<double> s_list = linspace(0, 1, 101);
  std::vector<double> l_list = linspace(0, 1, 6);
  std::vector<int> N_list = {2, 4, 8, 16, 32, 64, 128};
  for(double l : l_list)
  {
    std::vector<std::vector<double>> energies;
    for(int N: N_list)
    {
      std::cout << "Calculating " << N << " Spins for lambda "
		<< l << "." << std::endl;
      std::vector<double> energy;
      for(double s: s_list)
      {
	SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
	es = diagonalize(H0plusVaffplusVtf(N, s, l, p));
	double ev = es.eigenvalues()(0);
	energy.push_back(ev/N);
      }
      energies.push_back(energy);
    }
    std::cout << "Done." << std::endl;
    Gnuplot gp("Rescaled concurrence");

    std::ostringstream strs;
    strs << l;
    std::string str = strs.str();
    std::ostringstream s;
    s << "Rescaled concurrence for  lambda=" + str;
    auto title = s.str();
    gp.set_title(title);
    gp.set_xlabel("s");
    for(int i=0; i<N_list.size(); ++i)
    {
      std::ostringstream s2;
      s2 << N_list[i] << " Spins";
      gp.set_style("lines").plot_xy(s_list, energies[i], s2.str());
    }  
    gp.unset_smooth();
    gp.showonscreen();
    std::cout << "Press Enter to exit." << std::endl;
    // "read" for max linux, "pause" for windows
    std::system("read");
  }
}

void lambdaNotOneConcurrence(const int& p)
{
  // Calculates the rescaled concurrence for lambda!=1
  std::vector<double> s_list = linspace(0, 1, 501);
  std::vector<double> l_list = linspace(0.2, 1., 5);
    std::vector<int> N_list = {2, 4, 8, 16, 32, 62};
  for(double l : l_list)
  {
    std::ostringstream strs;
    strs << l;
    std::string str = strs.str();
    if(l==0.)
      continue;
    std::vector<std::vector<unsigned long long int> > pascal, temp;
    std::vector<std::vector<double> > concurrences;
    for(int N : N_list)
    {
      temp = pascalTriangle(pascal.size(), N);
      pascal.insert(pascal.end(), temp.begin(), temp.end());
      std::cout << "Calculating concurrence " << N << " Spins." << std::endl;
      std::vector<double> concurrence;
      for(double s: s_list)
      {
	SelfAdjointEigenSolver<Matrix<double, Dynamic, Dynamic> > es;
	es = diagonalize(H0plusVaffplusVtf(N, s, l, p));
	concurrence.push_back(calculateConcurrence(es, pascal, N)*(N-1));
      }
      concurrences.push_back(concurrence);
    }
    std::cout << "Done." << std::endl;
    Gnuplot gp("Rescaled concurrence Cr");
    std::ostringstream s, s2;
    s << "Rescaled Concurrece for p=" + std::to_string(p) + " and lambda=" << l;
    s2 << "cRforLambda" << l << "lines";
    auto title = s.str();
    auto title2 = s2.str();
    gp.set_title(title);
    gp.set_xlabel("s");
    gp.set_ylabel("Cr");
    for(int i=0; i<N_list.size(); ++i)
    {
      std::ostringstream s2;
      s2 << N_list[i] << " Spins";
      gp.set_style("points").plot_xy(s_list, concurrences[i], s2.str());
    }
    title = ("../results/concurrence/p" + std::to_string(p) + "/lambda" + str + "limit.csv");
    std::cout << "reading " << title << std::endl;
    std::vector<double> limit = readLimit(title);
    gp.savetops(title2);	  
    gp.set_style("lines").plot_xy(s_list, limit, "limit");
    gp.unset_smooth();
    gp.showonscreen();
    std::system("read");
  }
}

// Author Fredrik Jadebeck
//
// Source for the quantumMechanics.py port

#include <assert.h>
#include <cmath>
#include <Eigen/Core>
#include <iostream>
#include <set>
#include <vector>

#include "quantumMechanics.hpp"

using namespace Eigen;

std::vector<double> gen_m(const int& N)
{
  assert(N > 0);
  double S = (double)N/2.;
  std::vector<double> m;
  for(int i=0; i<N+1; ++i)
  {
    m.push_back(-S+i);
  }
  return m;
}

int kronecker(const double& m, const double& n)
{
  if(m==n)
  {
    return 1;
  }
  else
  {
    return 0;
  }
}

double mSxn(const double& m, const double& n, const double& S)
{
  assert(S > 0);
  return (kronecker(m, n+1) + kronecker(m+1, n))*0.5*std::sqrt(S*(S+1) -m*n);
}

double mSzn(const double& m, const double& n, const double& S)
{
  assert(S > 0);
  return kronecker(m, n)*m;
}

Matrix<double, Dynamic, Dynamic> Sx(const int& N)
{
  assert(N > 0);
  double S = (double)N/2;
  std::vector<double> m = gen_m(N);
  Matrix<double, Dynamic, Dynamic> Sx;
  Sx.resize(N+1, N+1);
  Sx.setZero();
  for(int i=0; i<N; ++i)
  {
    Sx(i, i+1) = mSxn(m[i], m[i+1], S);
    Sx(i+1, i) = mSxn(m[i+1], m[i], S);
  }
  return Sx;
}

Matrix<double, Dynamic, Dynamic> Sz(const int& N)
{
  assert(N > 0);
  double S = (double)N/2;
  std::vector<double> m = gen_m(N);
  Matrix<double, Dynamic, Dynamic> Sz;
  Sz.resize(N+1, N+1);
  Sz.setZero();
  for(int i=0; i<N+1; ++i)
  {
    Sz(i, i) = mSzn(m[i], m[i], S);
  }
  return Sz;
}

state::state(){}
state::state(std::string spins)
{
  this->quantumState.push_back(spins);
}
state::state(std::vector<std::string> spins)
{
  this->quantumState = spins;
}
state::state(const state& obj)
{
  this->quantumState = obj.quantumState;
}
double state::getNorm() const
{
  return std::sqrt(this->quantumState.size());
}
double state::scalar_product(const state& other) const
{
  double endResult = 0;
  for(auto i : this->quantumState)
    for(auto j : other.quantumState)
    {
      double result = 1;
      if(i!=j)
	result = 0;
      endResult += result;
    }
  return endResult/(this->getNorm()*other.getNorm());
}
ketBra state::tensor_product(const state& other) const
{
  assert(this->quantumState.size() == other.quantumState.size());
  return ketBra(*this, other);
}
state state::operator+(const state& other) const
{
  auto copy = *this;
  copy.quantumState.insert(std::end(copy.quantumState),
			   std::begin(other.quantumState),
			   std::end(other.quantumState));
  return copy;
}
bool state::operator==(const state& other) const
{
  bool result = false;
  if(std::abs(this->scalar_product(other)-1) < 0.00001)
    result = true;
  return result;
}
std::ostream& operator<<(std::ostream& os, const state& other)
{
  std::string ret;
  for(auto v: other.quantumState)
    ret += v;
  return os << ret;
};

ketBra::ketBra()
{
  this->amplitude = 0.;
}
ketBra::ketBra(const state& ket,
	       const state& bra)
{
  this->tket = ket;
  this->tbra = bra;
  this->amplitude = 1;
}
ketBra::ketBra(const state& ket, const state& bra, const double& amp) : ketBra(ket, bra)
{
  this->amplitude = amp;
}
ketBra::ketBra(const ketBra& obj)
{
  this->tket = obj.tket;
  this->tbra = obj.tbra;
  this->amplitude = obj.amplitude;
}
ketBra ketBra::multiply_with(const ketBra& other)
{
  auto result = this->tket.tensor_product(other.tbra);
  result.amplitude *= this->tbra.scalar_product(other.tket);
  return result;
}
bool ketBra::operator==(const ketBra& other)
{
  bool result = false;
  if(this->tket == other.tket and this->tbra == other.tbra)
    result = true;
  else if(this->amplitude == 0. and other.amplitude == 0.)
    result = true;
  return result;
}
std::ostream& operator<<(std::ostream& os, const ketBra& other)
{
  if(other.amplitude == 1)
  {
    return os << "|" << other.tket << "><" << other.tbra <<"|";
  }
  else if(other.amplitude != 0.)
  {
    return os << other.amplitude << "|" << other.tket << "><" << other.tbra <<"|";
  }
  else
    return os << 0;
}


dm::dm(Matrix<double, Dynamic, Dynamic> rho,
       std::vector<Matrix<double, Dynamic, Dynamic> > zeemanBasis,
       const int& N)
{
  this->N=N;
}
dm::dm(const dm& obj)
{
  // Todo Implement copy constructor after more details are kown
  // about this class.
}
Matrix<double, Dynamic, Dynamic> dm::nparray(){}
dm dm::ptrace(const int& k){}

void zeeman(){}

// Author Johann Fredrik Jadebeck
#include <assert.h>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

#include "/home/fredrik/repos/gnuplot-cpp/gnuplot_i.hpp"

using namespace std;

double f(const double& mx, const double& mz, const double& s, const double& l, const double& p)
{
  // todo insert free energy formula
}

double m_x(const double& mx, const double& mz, const double& s, const double& l, const double& p)
{
  if(s<1./(3-2*l))
  {
    return ((1-s-2*s*(1-l)*mx) /
  	    std::sqrt(std::pow((p*s*l*std::pow(mz, p-1)), 2) + std::pow((1-s-2*s*(1-l)*mx), 2)));
  }
  else
    return ((1-s-2*s*(1-l)*mx) /
	    std::sqrt(std::pow((p*s*l*std::pow(mz, p-1)), 2) + std::pow((1-s-2*s*(1-l)*mx), 2)));
}

double sign_mx(const double& mx, const double& mz, const double& s, const double& l, const double& p)
{
  if(s<1./(3-2*l))
  {
    return ((1-s-2*s*(1-l)*mx) /
  	    std::sqrt(std::pow((p*s*l*std::pow(mz, p-1)), 2) + std::pow((1-s-2*s*(1-l)*mx), 2)) - mx);
  }
  else
    return ((1-s-2*s*(1-l)*mx) /
	    std::sqrt(std::pow((p*s*l*std::pow(mz, p-1)), 2) + std::pow((1-s-2*s*(1-l)*mx), 2)) - mx);
}

double m_z(const double& mx, const double& mz, const double& s, const double& l, const double& p)
{
  // if(s<1./(3-2*l))
  // {
  //   return ((p*s*l*std::pow(mz, p-1)) /
  // 	    std::sqrt(std::pow((p*s*l*std::pow(mz, p-1)), 2) + std::pow((1-s-2*s*(1-l)*mx), 2)));
  // }
  // else
  return ((p*s*l*std::pow(mz, p-1)) /
	  std::sqrt(std::pow((p*s*l*std::pow(mz, p-1)), 2) + std::pow((1-s-2*s*(1-l)*mx), 2)));

	  
}

template<typename T>
std::vector<double> linspace(const T& s, const T& e, const int& n)
{
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

std::vector<std::vector<double>> create_const(const int& x, const int& y, const double& c)
{
  std::vector<std::vector<double>> one(x);
  for(auto &k : one)
  {
    k = std::vector<double>(y, c);
  }
  return one;
}

// Note ofstream has no copy constructor, we need to pass a reference for it to make sense!
auto run(const int& ns, const int& nl, const int&p, const double& precision, ofstream & file)
{
  file << "lambda" << "," << "s" << "," << "mx"
       << "," << "mz" << ","
       << "mz^2+mx^2" << "," << "tries"
       << std::endl;

  std::vector<double> sspace = linspace(0, 1, ns);
  std::vector<double> lspace = linspace(0.1, 1., nl);
  std::vector<std::vector<double>> mx = create_const(nl, ns, 0.8);
  std::vector<std::vector<double>> mz = create_const(nl, ns, 0.6);
  double prevmx, prevmz;
  for(int i=0; i<nl; ++i)
  {
    for(int j=0; j<ns; ++j)
    {
      double m = 0;
      do
      {
	++m;
	prevmx = mx[i][j];
	prevmz = mz[i][j];
	mx[i][j] = (m_x(prevmx, prevmz, sspace[j], lspace[i], p));
	// Uses this formula because mz has two solutions, one of which is wrong
	mz[i][j] = 1 - mx[i][j]*mx[i][j];
	if(m>1000)
	{
	  auto testmx = linspace(0, 1, 1001);
	  for(auto &k : testmx)
	  {
	    auto temp_mz = 1-k*k;
	    auto sign = sign_mx(k, temp_mz, sspace[j], lspace[i], p);
	    if(sign < 0)
	    {
	      mx[i][j] = k;
	      mz[i][j] = temp_mz;
	      break;
	    }
	  }
	  break;
	}
      }
      while((std::abs(prevmz-mz[i][j]) > precision) or (std::abs(prevmx - mx[i][j]) > precision));
      if(mx[i][j] < 0.)
	mx[i][j] = mx[i][j]*(-1);
      file << lspace[i] << "," << sspace[j] << "," << mx[i][j]
		<< "," << mz[i][j] << ","
		<< mx[i][j]*mx[i][j] + mz[i][j]*mz[i][j] << "," << m
	   << std::endl;
      cout << lspace[i] << "," << sspace[j] << "," << mx[i][j]
	   << "," << mz[i][j] << ","
	   << mx[i][j]*mx[i][j] + mz[i][j]*mz[i][j] << "," << m
	   << std::endl;
	    
      //assert(mx[i][j]*mx[i][j] + mz[i][j]*mz[i][j] <= 1 + precision);
      //assert(mx[i][j]*mx[i][j] + mz[i][j]*mz[i][j] >= 1 - precision);
    }
  }
  auto f = mx;
  return std::make_tuple(mx, mz, f);
}

int main(int argc, char** argv)
{
  int ns = 10001;
  int nl = 11;
  int p = 5;
  double precision = 1e-14;
  ofstream myfile;
  myfile.open("result.csv");
  myfile << "Thermodynamic Limit Magnetization\n";
  myfile << "mx,mz,f\n";
  auto result_tuple = run(ns, nl, p, precision, myfile);
  std::vector<std::vector<double>> mx = std::get<0>(result_tuple);
  std::vector<std::vector<double>> mz = std::get<1>(result_tuple);
  std::vector<std::vector<double>> f = std::get<2>(result_tuple);
  // for(int i=0; i < mx.size(); ++i)
  //   {
  //     myfile << mx[i] << "," << mz[i] << "," << ",";
  //     myfile << "\n";
  //   }
  myfile.close();
  std::vector<double> sspace = linspace(0, 1, ns);
  std::vector<double> lspace = linspace(0, 1, nl);
  std::vector<double> y;
  y.push_back(0.);
  y.push_back(1.);
  
  for(int i=0; i < mx.size(); ++i)
  {
    std::vector<double> x;
    x.push_back(1./(3-2*lspace[i]));
    x.push_back(1./(3-2*lspace[i]));
    std::vector<double> QP2;
    std::vector<double> sQP2;
    for(const auto& s:sspace)
    {
      if(s>1/(3-2*lspace[i]))
      {
	sQP2.push_back(s);
	QP2.push_back((1.-s)/(2*s*(1.-lspace[i])));
      }
    }
    Gnuplot gp("Magnetization in X direction");
    std::ostringstream s;
    s << "Magnetization for lambda " << lspace[i];
    auto title = s.str();
    gp.set_title(title);
    gp.set_xlabel("s");
    gp.set_style("points").plot_xy(sspace, mx[i], "m_x");
    gp.set_style("points").plot_xy(sspace, mz[i], "m_z");
    // gp.set_style("points").plot_xy(sQP2, QP2, "m_xQP2");
    gp.set_smooth("bezier").set_style("line").plot_xy(x, y, "boundary QP1 to QP2");
    gp.unset_smooth();
    gp.showonscreen();
    cin.get();
  }
}

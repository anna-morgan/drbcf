#ifndef RNG_H
#define RNG_H

#include "arma_config.h"
#include <RcppArmadillo.h>
#include "gig.h"

using std::vector;

class RNG
{
 private:
 public:
  // Continuous Distributions
  double gig(double lambda, double chi, double psi) {
    return rgig(lambda, chi, psi);
  }
  double uniform(double x = 0.0, double y = 1.0)
    { return R::runif(x, y); }
  double normal(double mu = 0.0, double sd = 1.0)
    { return R::rnorm(mu, sd); }
  double gamma(double shape = 1, double scale = 1)
  { return R::rgamma(shape, 1)*scale; }
  double chi_square(double df)
    { return R::rchisq(df); }//return gamma(df / 2.0, 0.5); }
  double beta(double a1, double a2)
    { const double x1 = gamma(a1, 1); return (x1 / (x1 + gamma(a2, 1))); }

  void uniform(vector<double>& res, double x = 0.0, double y = 1.0) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = uniform(x, y);
  }
  void normal(vector<double>& res, double mu = 0.0, double sd = 1.0) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = normal(mu, sd);
  }
  void gamma(vector<double>& res, double shape = 1, double scale = 1) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = gamma(shape, scale);
  }
  void chi_square(vector<double>& res, double df) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = chi_square(df);
  }
  void beta(vector<double>& res, double a1, double a2) {
    for (vector<double>::iterator i = res.begin(); i != res.end(); ++i)
      *i = beta(a1, a2);
  }

    //sample from N(Phi^(-1)m, Phi^(-1))
    arma::vec rmvnorm_post(arma::vec &m, arma::mat &Phi);

   // sample from two-sided truncated normal
    double rtnormlohi(double mean, double lo, double hi) {
      double F_a = R::pnorm(lo, mean, 1, true, false);
      double F_b = R::pnorm(hi, mean, 1, true, false);
      double u = R::runif(F_a, F_b);
      double x = R::qnorm(u, mean, 1, true, false);
      return(x);
    }
    
    // Generate random MVT vector
    arma::vec rmvt_cpp(arma::mat Sigma, // scale matrix
                       arma::vec delta, // noncentrality parameters
                       int df){ 
      arma::rowvec delta_r = arma::conv_to<arma::rowvec>::from(delta);
      int ncols = Sigma.n_cols;
      arma::mat Y = arma::randn(1, ncols);
      Y = Y * arma::chol(Sigma);
      double u = R::rchisq(df);
      return(arma::conv_to<arma::vec>::from(delta_r + sqrt(df / u) * Y));
    }
    
}; // class RNG


inline double znorm() {
  return R::rnorm(0.0, 1.0);
}


template<class T>
  int rdisc_log(T &logweights, int n=-1) {
    if(n==-1) n = logweights.size();
    typename T::iterator itb = logweights.begin();
    typename T::iterator ite = logweights.begin() + n;

    double m = *std::max_element(itb, ite);
    double s = 0;
    std::vector<double> weights(n, 0.0);
    for(int i=0; i<n; ++i) {
      weights[i] = exp(logweights[i] - m);
      s += weights[i];
    }
    double u = s*R::runif(0.0, 1.0);
    double cs = weights[0];
    int i=0;
    while((u>cs) & (i<n)) {
      ++i;
      cs += weights[i];
    }
    return i;
  }


template<class T>
  int rdisc_normed(T &probs, int n=-1) {
    if(n==-1) n = probs.size();
    double u = R::runif(0.0, 1.0);
    double cs = probs[0];
    int i=0;
    while((u>cs) & (i<n)) {
      ++i;
      cs += probs[i];
    }
    return i;
  }


template<class T>
  int rdisc_log_inplace(T &logweights, int n=-1, double u=-1.0) {
    if(n==-1) n = logweights.size();
    if(u==-1) u = R::runif(0.0, 1.0);
    typename T::iterator itb = logweights.begin();
    typename T::iterator ite = logweights.begin() + n;

    double m = *std::max_element(itb, ite);
    double s = 0;
    //vector<double> weights(n, 0.0);
    for(int i=0; i<n; ++i) {
      logweights[i] = exp(logweights[i] - m);
      s += logweights[i];
    }
    u = s*u;
    double cs = logweights[0];
    int i=0;
    while((u>cs) & (i<n)) {
      ++i;
      cs += logweights[i];
    }
    return i;
  }

double rtnormlo0(double lo);
double rtnormlo1(double mean, double lo);
double rtnormhi1(double mean, double lo);
double rtnormlo(double mean, double sd, double lo);
arma::vec rmvnorm_post(arma::vec &m, arma::mat &Phi);
#endif // RNG_H


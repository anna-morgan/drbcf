#include "arma_config.h"
#include <RcppArmadillo.h>

#include <cmath>
#include "funs.h"
#include "rng.h"
#include <map>
#ifdef MPIBART
#include "mpi.h"
#endif

#include "slice.h"

using Rcpp::Rcout;
using namespace arma;
using namespace Rcpp;


arma::sp_mat spdiag_from_vec(arma::vec X) {
  arma::sp_mat ret(X.size(), X.size());
  for(size_t i=0; i<X.size(); ++i) {
    ret(i,i) = X(i);
  }
  return(ret);
}

arma::sp_mat dummy_to_sparse(arma::mat X) {
  arma::sp_mat ret(X.n_rows, X.n_cols);
  for(size_t i=0; i<X.n_rows; ++i) {
    for(size_t j=0; j<X.n_cols; ++j) {
      if(X(i,j)>0.1) ret(i,j) = 1;
    }
  }
  return(ret);
}

arma::sp_vec dummy_to_sparse(arma::vec X) {
  arma::sp_vec ret(X.size());
  for(size_t i=0; i<X.size(); ++i) {
    if(X(i)>0.1) ret(i) = 1;
  }
  return(ret);
}


//find variables n can split on, put their indices in goodvars
void getbadvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& badvars)
{
  int L,U;
  for(size_t v=0;v!=xi.size();v++) {//try each variable
    L=0; U = xi[v].size()-1;
    n->rg(v,&L,&U);
    if(U<L) badvars.push_back(v);
  }
}


void update_dart_noaug(std::vector<tree>& trees, 
                       std::vector<int> &vargrps,
                       std::vector<double> &varwts,
                       int &uv,
                       pinfo& pi, dinfo& di, xinfo& xi, RNG& gen) {
  std::vector<double> use_counts(di.p, 0);
  std::vector<double> var_use_counts(uv, 0);


  //Rcpp::Rcout << "counts" << endl;
  for(size_t j=0;j<trees.size();j++) {
    std::vector<tree*> tj_nodes;
    trees[j].getnobots(tj_nodes);
    for(size_t b=0; b<tj_nodes.size(); ++b) {
      use_counts[tj_nodes[b]->getv()] += 1.0;
    }
  }
  
  
  //Rcpp::Rcout << "agg counts" << endl;
  if(uv == di.p) { //no categorical vars
    var_use_counts = use_counts;
  } else {
    for(size_t j=0; j<use_counts.size(); ++j) {
      var_use_counts[vargrps[j]] += use_counts[j]; //Aggregate counts for dummies
    }
  }

  //Rcpp::Rcout << "agg p sample" << endl;
  double tot = 0;
  for(size_t v = 0; v < pi.var_probs_short.size(); ++v) {
    pi.var_probs_short[v] = R::rgamma(var_use_counts[v] + pi.alpha/(di.p), 1);
    tot += pi.var_probs_short[v];
  }
  for(size_t v = 0; v < pi.var_probs_short.size(); ++v) pi.var_probs_short[v] /= tot;
  
  //Rcpp::Rcout << "dis agg p" << endl;
  if(uv == di.p) { //no categorical vars
    pi.var_probs = pi.var_probs_short;
  } else {
    for(size_t j=0; j<pi.var_probs.size(); ++j) {
      //var_use_counts[vargrps[j]] += use_counts[j]; //Aggregate counts for dummies
      pi.var_probs[j] = pi.var_probs_short[vargrps[j]]*varwts[j];
      //Rcpp::Rcout << pi.var_probs[j] << " ";
    }
    //Rcpp::Rcout << endl;
  }
  
  //Rcpp::Rcout << "slice" << endl;
  
  /*
  Rcpp::Rcout << "counts" << endl;
  for(size_t i=0; i<var_use_counts.size(); ++i) Rcpp::Rcout << var_use_counts[i] << endl;
  
  Rcpp::Rcout << "probs" << endl;
  for(size_t i=0; i<pi.var_probs_short.size(); ++i) Rcpp::Rcout << pi.var_probs_short[i] << endl;
  
   */
  
  
  ld_dart_alpha logdens(var_use_counts, pi.var_probs_short);
  
  double a0 = pi.dart_alpha;
  pi.dart_alpha = slice(a0, &logdens, 1.0, INFINITY, 0.0, INFINITY);
  
  
  //Rcpp::Rcout << "end slice" << endl;

}


void update_dart(std::vector<tree>& trees, pinfo& pi, dinfo& di, xinfo& xi, RNG gen) {
  std::vector<double> var_use_counts(di.p, 0);
  
  for(size_t j=0;j<trees.size();j++) {
    std::vector<tree*> tj_nodes;
    trees[j].getnobots(tj_nodes);
    for(size_t b=0; b<tj_nodes.size(); ++b) {
      var_use_counts[tj_nodes[b]->getv()] += 1.0;
      std::vector<size_t> badvars; //variables this tree couldn't split on
      getbadvars(tj_nodes[b], xi, badvars);
      if(badvars.size()>0) {
        double total_prob = 0;
        vector<double> sample_probs(badvars.size());
        for(size_t r = 0; r<badvars.size(); ++r) {
          total_prob += pi.var_probs[badvars[r]];
          sample_probs[r] = pi.var_probs[badvars[r]];
        }
        for(size_t r = 0; r<badvars.size(); ++r) sample_probs[r]/=total_prob;
        double numaug = R::rnbinom(1, 1-total_prob);
        int tt = 0;
        while( (numaug>0.5) & (tt<50000) ) {
          var_use_counts[badvars[rdisc(&sample_probs[0], gen)]] += 1;
          numaug -= 1; tt += 1;
        }
      }
    }
  }

  double tot = 0;
  for(size_t v = 0; v < di.p; ++v) {
    pi.var_probs[v] = R::rgamma(var_use_counts[v] + pi.alpha/(di.p * pi.var_sizes[v]), 1);
    tot += pi.var_probs[v];
  }
  for(size_t v = 0; v < di.p; ++v) pi.var_probs[v] /= tot;

  ld_dart_alpha logdens(var_use_counts, pi.var_probs);

  double a0 = pi.dart_alpha;
  pi.dart_alpha = slice(a0, &logdens, 1.0, INFINITY, 0.0, INFINITY);

}

void update_scale(double* r, double* fits, size_t n, double sigma, pinfo& pi, RNG& gen) {

  // assumes that the scale multiplier is contained in fits!

  double a = 0.0; double b = 0.0;
  //double sigma = pi.sigma*fabs(pi.eta);
  for(size_t k=0;k<n;k++) {
    //double r = y[k] - allfit_con[k];
    //if(randeff) r -= allfit_random[k];

    a += fits[k] * r[k]/pi.eta;
    b += fits[k] * fits[k]/(pi.eta*pi.eta);
  }

  double var = 1 / (1 / (pi.gamma * pi.gamma) + b/(sigma*sigma));
  double mean = var * (0.0/(pi.gamma*pi.gamma) + a/(sigma*sigma));

  pi.eta = mean + gen.normal(0.,1.) * sqrt(var);

  if(pi.scale_df>0) {
    pi.gamma = sqrt(0.5*(pi.scale_df+pi.eta*pi.eta)/gen.gamma((1.0 + pi.scale_df)/2.0, 1.0));
  }

}

//todo: make this a template or change types of allfits elsewhere
void update_scale(double* r, std::vector<double> &fits, size_t n, double sigma, pinfo& pi, RNG& gen) {
  
  // assumes that the scale multiplier is contained in fits!
  
  double a = 0.0; double b = 0.0;
  //double sigma = pi.sigma*fabs(pi.eta);
  for(size_t k=0;k<n;k++) {
    //double r = y[k] - allfit_con[k];
    //if(randeff) r -= allfit_random[k];
    
    a += fits[k] * r[k]/pi.eta;
    b += fits[k] * fits[k]/(pi.eta*pi.eta);
  }
  
  double var = 1 / (1 / (pi.gamma * pi.gamma) + b/(sigma*sigma));
  double mean = var * (0.0/(pi.gamma*pi.gamma) + a/(sigma*sigma));
  
  pi.eta = mean + gen.normal(0.,1.) * sqrt(var);
  
  if(pi.scale_df>0) {
    pi.gamma = sqrt(0.5*(pi.scale_df+pi.eta*pi.eta)/gen.gamma((1.0 + pi.scale_df)/2.0, 1.0));
  }
  
}

//-------------------------------------------------------------
// Generates realizations from multivariate normal.
//-------------------------------------------------------------
mat rmvnormArma(int n, vec mu, mat sigma) {
   //-------------------------------------------------------------
   // INPUTS:	   n = sample size
   //				   mu = vector of means
   //				   sigma = covariance matrix
   //-------------------------------------------------------------
   // OUTPUT:	n realizations of the specified MVN.
   //-------------------------------------------------------------
   int ncols = sigma.n_cols;
   mat Y = randn(n, ncols);
   mat result = (repmat(mu, 1, n).t() + Y * chol(sigma)).t();
   return result;
}

struct cmpdouble {
  bool operator()(const double &a, const double &b) const {
    return fabs(a-b)<1e-8;
  }
};

//Basis

void allsuff_basis(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv, std::vector<sinfo>& sv, 
                   std::vector<tree::tree_cp>& node_pointers)
{
  // Bottom nodes are written to bnv.
  // Suff stats for each bottom node are written to elements (each of class sinfo) of sv.
  // Initialize data structures
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations.  tree_cp bc not modifying tree directly.
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  double y;          //current y
  double t;          //current t
  double omega;

  bnv.clear();      // Clear the bnv variable if any value is already saved there.
  x.getbots(bnv);   // Save bottom nodes for x to bnv variable.

  typedef tree::npv::size_type bvsz;  
  bvsz nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.

  // Resize vectors within sufficient stats to have di.tlen length.
  for(size_t i = 0; i < nb; ++i){
    //sv[i].n_vec.resize(di.tlen);
    sv[i].sy = 0;
    sv[i].n0 = 0.0;
    sv[i].n = 0.0;
    sv[i].n_unique = 0;
    sv[i].sy_vec.zeros(di.basis_dim);
    sv[i].WtW.zeros(di.basis_dim, di.basis_dim);
  }
  
  std::vector<std::map<int, int> > unique_groups(sv.size());

  // bnmap is a tuple (lookups, like in Python).  Want to index by bottom nodes.
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;  // bnv[i]
  //map looks like
  // bottom node 1 ------ 1
  // bottom node 2 ------ 2

  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;  //Index value: di.x is pointer to first element of n*p data vector.  Iterates through each element.
    y = di.y[i];           // Resolves to r.

    //tbn = x.bn(xx,xi); // Find bottom node for this observation.
    tbn = node_pointers[i];
    ni = bnmap[tbn];   // Map bottom node to integer index

    ++(sv[ni].n);
    if(di.group) unique_groups[ni][di.groups[i]] += 1;

    if(di.basis_dim == 1) {
      omega = di.omega[i];
      sv[ni].sy += omega*y;
      sv[ni].n0 += omega*omega;
    } else {

      //get design vector
      double *omega_i_tmp = di.omega + i*di.basis_dim;
      arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);

      sv[ni].sy_vec += y*omega_i;

      for(size_t j=0; j<di.basis_dim; ++j) {
        sv[ni].WtW.at(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
        for(size_t g=0; g<j; ++g) {
          double a = omega_i_tmp[j]*omega_i_tmp[g];
          sv[ni].WtW.at(g,j) += a;
          //sv[ni].WtW(j,g) += a; //this is faster than below?
          //sv[ni].WtW[i,j] = sv[ni].WtW[i,j] + a;
          //sv[ni].WtW[j,i] = sv[ni].WtW[j,i] + a; //get this outside obs loop later
        }
      }
    }

  }
  
  
  for(size_t q=0; q<sv.size(); ++q) {
    if(di.group) {   
      sv[q].n_unique = unique_groups[q].size();
    } else {
      sv[q].n_unique = sv[q].n;
    }
  }

  if(di.basis_dim>1) {
    for(size_t q=0; q<sv.size(); ++q) {
      for(size_t j=0; j<di.basis_dim; ++j) {
        for(size_t g=0; g<j; ++g) {
          sv[q].WtW.at(j,g) = sv[q].WtW.at(g,j);
        }
      }
    }
  }

  if(di.basis_dim<2) {
    for(size_t j=0; j<sv.size(); ++j) {
      sv[j].WtW.at(0,0) = sv[j].n0;
      sv[j].sy_vec.at(0) = sv[j].sy;
    }
  }


}

void allsuff_basis_birth(tree& x, tree::tree_cp nx, size_t v, size_t c, xinfo& xi, dinfo& di, tree::npv& bnv,
                         std::vector<sinfo>& sv, sinfo& sl, sinfo& sr, std::vector<tree::tree_cp>& node_pointers,
                         bool split)
{
  // Bottom nodes are written to bnv.
  // Suff stats for each bottom node are written to elements (each of class sinfo) of sv.
  // Initialize data structures
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations.  tree_cp bc not modifying tree directly.
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  double y;          //current y
  double t;          //current t
  double omega;

  bnv.clear();      // Clear the bnv variable if any value is already saved there.
  x.getbots(bnv);   // Save bottom nodes for x to bnv variable.

  typedef tree::npv::size_type bvsz;
  bvsz nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.

  // Resize vectors within sufficient stats to have di.tlen length.
  for(size_t i = 0; i < nb; ++i){
    sv[i].sy = 0;
    sv[i].n0 = 0.0;
    sv[i].n = 0;
    sv[i].n_unique = 0;
    sv[i].sy_vec.zeros(di.basis_dim);
    sv[i].WtW.zeros(di.basis_dim, di.basis_dim);
  }
  
  std::vector<std::map<int, int> > unique_groups(sv.size());
  std::vector<std::map<int, int> > unique_groups_lr(2); //1st elt is l node, second is r node

  // bnmap is a tuple (lookups, like in Python).  Want to index by bottom nodes.
  std::map<tree::tree_cp,size_t> bnmap;
  for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;  // bnv[i]
  //map looks like
  // bottom node 1 ------ 1
  // bottom node 2 ------ 2

  double *omega_i_tmp;
  arma::vec omega_i;
  bool in_candidate_nog, left, right;

  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    y = di.y[i];

    //tbn = x.bn(xx,xi); // Find bottom node for this observation.
    tbn = node_pointers[i];
    ni = bnmap[tbn];   // Map bottom node to integer index

    left = false; right = false;

    in_candidate_nog = (tbn == nx) & split;
    if(in_candidate_nog) {
      left  = (xx[v] < xi[v][c]);
      right = !(xx[v] < xi[v][c]);
    }

    ++(sv[ni].n);
    if(di.group) unique_groups[ni][di.groups[i]] += 1;
    
    if(left) {
      sl.n += 1;
      if(di.group) unique_groups_lr[0][di.groups[i]] += 1;
    }
    if(right) {
      sr.n += 1;
      if(di.group) unique_groups_lr[1][di.groups[i]] += 1;
    }

    if(di.basis_dim == 1) {
      omega = di.omega[i];
      sv[ni].sy += omega*y;
      sv[ni].n0 += omega*omega;
      if(left) {
        sl.sy += omega*y;
        sl.n0 += omega*omega;
      }
      if(right) {
        sr.sy += omega*y;
        sr.n0 += omega*omega;
      }

    } else {
      omega_i_tmp = di.omega + i*di.basis_dim;
      //get design vector
      arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);

      sv[ni].sy_vec += y*omega_i;
      if(left) sl.sy_vec += y*omega_i;
      if(right) sr.sy_vec += y*omega_i;

      for(size_t j=0; j<di.basis_dim; ++j) {
        double a = omega_i_tmp[j]*omega_i_tmp[j];
        sv[ni].WtW.at(j,j) += a;
        if(left) sl.WtW.at(j,j) += a;
        if(right) sr.WtW.at(j,j) += a;
        for(size_t g=0; g<j; ++g) {
          double a = omega_i_tmp[j]*omega_i_tmp[g];
          sv[ni].WtW.at(g,j) += a;
          if(left) sl.WtW.at(g,j) += a;
          if(right) sr.WtW.at(g,j) += a;
        }
      }
    }
  }
  
  for(size_t q=0; q<sv.size(); ++q) {
    if(di.group) {   
      sv[q].n_unique = unique_groups[q].size();
    } else {
      sv[q].n_unique = sv[q].n;
    }
  }

  if(di.group) {   
    sl.n_unique = unique_groups_lr[0].size();
    sr.n_unique = unique_groups_lr[1].size();
  } else {
    sl.n_unique = sl.n;
    sr.n_unique = sr.n;
  }
  
  
  
  if(di.basis_dim>1) {

    //Rcout << "L" << endl << sl.WtW << endl << sl.sy_vec << endl;
    //Rcout << "R" << endl << sr.WtW << endl << sr.sy_vec << endl;

    for(size_t q=0; q<sv.size(); ++q) {
      for(size_t j=0; j<di.basis_dim; ++j) {
        for(size_t g=0; g<j; ++g) {
          sv[q].WtW(j,g) = sv[q].WtW(g,j);
          sl.WtW(j,g) = sl.WtW(g,j);
          sr.WtW(j,g) = sr.WtW(g,j);
        }
      }
    }

    //Rcoutt << "L" << endl << sl.WtW << endl;
    //Rcoutt << "R" << endl << sr.WtW << endl;
  }

  if(di.basis_dim<2) {
    sl.WtW.at(0,0) = sl.n0;
    sl.sy_vec.at(0) = sl.sy;
    sr.WtW.at(0,0) = sr.n0;
    sr.sy_vec.at(0) = sr.sy;
    for(size_t j=0; j<sv.size(); ++j) {
      sv[j].WtW.at(0,0) = sv[j].n0;
      sv[j].sy_vec.at(0) = sv[j].sy;
    }
  }




}

void getsuff_basis(tree& x, tree::tree_cp nx, size_t v, size_t c, xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr)
{
  double *xx;//current x
  double y;  //current y
  double t;  //current t

  sl.n=0;sl.sy=0.0;sl.sy2=0.0;
  sr.n=0;sr.sy=0.0;sr.sy2=0.0;

  sl.WtW = zeros(di.basis_dim,di.basis_dim); sl.sy_vec = zeros(di.basis_dim);
  sr.WtW = zeros(di.basis_dim,di.basis_dim); sr.sy_vec = zeros(di.basis_dim);

  bool orig = false;

  double omega;

  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;


    if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node

      y = di.y[i];   // extract current yi.  resolves to r.

      //get design vector
      double *omega_i_tmp = di.omega + i*di.basis_dim;
      arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);

      if(xx[v] < xi[v][c]) { // Update left.

        ++(sl.n);

        if(di.basis_dim==1) {
          omega = di.omega[i];
          sl.sy += omega*y;
          sl.n0 += omega*omega;

        } else {

          sl.sy_vec += y*omega_i;
          for(size_t j=0; j<di.basis_dim; ++j) {
            //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
            sl.WtW(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
            for(size_t i=0; i<j; ++i) {
              double a = omega_i_tmp[j]*omega_i_tmp[i];
              sl.WtW[i,j] = sl.WtW[i,j] + a;
              //sl.WtW(j,i) += a;
            }
          }

        }


      } else { //Update right.

        ++(sr.n);

        if(di.basis_dim==1) {
          omega = di.omega[i];
          sr.sy += omega*y;
          sr.n0 += omega*omega;

        } else {

          sr.sy_vec += y*omega_i;
          for(size_t j=0; j<di.basis_dim; ++j) {
            //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
            sr.WtW(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
            for(size_t i=0; i<j; ++i) {
              double a = omega_i_tmp[j]*omega_i_tmp[i];
              sr.WtW[i,j] = sr.WtW[i,j] + a;
              //sr.WtW(j,i) += a;
            }
          }

        }


      }
    }
  }


  if(di.basis_dim<2) {
    sl.WtW[0,0] = sl.n0;
    sl.sy_vec[0] = sl.sy;
    sr.WtW[0,0] = sr.n0;
    sr.sy_vec[0] = sr.sy;
  }

  // this is faster without bounds checking, but incrementing w/ +=above is faster *with* bounds checking?
  for(size_t j=0; j<di.basis_dim; ++j) {
    //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
    for(size_t i=0; i<j; ++i) {
      sl.WtW[j,i] = sl.WtW[i,j];
      sr.WtW[j,i] = sr.WtW[i,j];
    }
  }

}

/*
void getsuff_basis(tree& x, tree::tree_cp nx, size_t v, size_t c, xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr)
{
  double *xx;//current x
  double y;  //current y
  double t;  //current t

  sl.n=0;sl.sy=0.0;sl.sy2=0.0;
  sr.n=0;sr.sy=0.0;sr.sy2=0.0;

  sl.WtW = zeros(di.basis_dim,di.basis_dim); sl.sy_vec = zeros(di.basis_dim);
  sr.WtW = zeros(di.basis_dim,di.basis_dim); sr.sy_vec = zeros(di.basis_dim);

  bool orig = false;

  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;


    if(nx==x.bn(xx,xi)) { //does the bottom node = xx's bottom node

      y = di.y[i];   // extract current yi.  resolves to r.

      //get design vector
      double *omega_i_tmp = di.omega + i*di.basis_dim;
      arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);


      if(xx[v] < xi[v][c]) { // Update left.

        ++(sl.n);
        sl.sy_vec += y*omega_i;

        if(orig) {
          sl.WtW += omega_i*omega_i.t();
        } else {
          for(size_t j=0; j<di.basis_dim; ++j) {
            //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
            sl.WtW(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
            for(size_t i=0; i<j; ++i) {
              double a = omega_i_tmp[j]*omega_i_tmp[i];
              sl.WtW[i,j] = sl.WtW[i,j] + a;
              //sl.WtW(j,i) += a;
            }
          }
        }



      } else { //Update right.
        ++(sr.n);
        sr.sy_vec += y*omega_i;

        if(orig) {
          sr.WtW += omega_i*omega_i.t();
        } else {
          for(size_t j=0; j<di.basis_dim; ++j) {
            //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
            sr.WtW(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
            for(size_t i=0; i<j; ++i) {
              double a = omega_i_tmp[j]*omega_i_tmp[i];
              sr.WtW[i,j] = sr.WtW[i,j] + a;
              //sr.WtW(j,i) += a;
            }
          }
        }

      }
    }
  }

  // this is faster without bounds checking, but incrementing w/ +=above is faster *with* bounds checking
  for(size_t j=0; j<di.basis_dim; ++j) {
    //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
    for(size_t i=0; i<j; ++i) {
      sl.WtW[j,i] = sl.WtW[i,j];
      sr.WtW[j,i] = sr.WtW[i,j];
    }
  }

}
*/
void getsuff_basis(tree& x, tree::tree_cp nl, tree::tree_cp nr, xinfo& xi, dinfo& di, sinfo& sl, sinfo& sr)
{
  double *xx;//current x
  double y;  //current y
  double t;  //current t

  bool orig = false;

  sl.n=0;sl.sy=0.0;sl.sy2=0.0;
  sr.n=0;sr.sy=0.0;sr.sy2=0.0;

  double omega;

  sl.WtW = zeros(di.basis_dim,di.basis_dim); sl.sy_vec = zeros(di.basis_dim);
  sr.WtW = zeros(di.basis_dim,di.basis_dim); sr.sy_vec = zeros(di.basis_dim);

  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    tree::tree_cp bn = x.bn(xx,xi);

    y = di.y[i];   // extract current yi.

    if(bn==nl) {

      //get design vector
      double *omega_i_tmp = di.omega + i*di.basis_dim;
      arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);

      ++(sl.n);

      if(di.basis_dim==1) {
        omega = di.omega[i];
        sl.sy += omega*y;
        sl.n0 += omega*omega;

      } else {

        sl.sy_vec += y*omega_i;
        for(size_t j=0; j<di.basis_dim; ++j) {
          //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
          sl.WtW(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
          for(size_t i=0; i<j; ++i) {
            double a = omega_i_tmp[j]*omega_i_tmp[i];
            sl.WtW[i,j] = sl.WtW[i,j] + a;
            //sl.WtW(j,i) += a;
          }
        }

      }

    }

    if(bn==nr) {

      //get design vector
      double *omega_i_tmp = di.omega + i*di.basis_dim;
      arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);

      ++(sr.n);

      if(di.basis_dim==1) {
        omega = di.omega[i];
        sr.sy += omega*y;
        sr.n0 += omega*omega;

      } else {

        sr.sy_vec += y*omega_i;
        for(size_t j=0; j<di.basis_dim; ++j) {
          //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
          sr.WtW(j,j) += omega_i_tmp[j]*omega_i_tmp[j];
          for(size_t i=0; i<j; ++i) {
            double a = omega_i_tmp[j]*omega_i_tmp[i];
            sr.WtW[i,j] = sr.WtW[i,j] + a;
            //sr.WtW(j,i) += a;
          }
        }

      }

    }
  }


  if(di.basis_dim<2) {
    sl.WtW[0,0] = sl.n0;
    sl.sy_vec[0] = sl.sy;
    sr.WtW[0,0] = sr.n0;
    sr.sy_vec[0] = sr.sy;
  }

  for(size_t j=0; j<di.basis_dim; ++j) {
    //sv[ni].sy_vec(j) += y*omega_i_tmp[j];
    for(size_t i=0; i<j; ++i) {
      sl.WtW[j,i] = sl.WtW[i,j];
      sr.WtW[j,i] = sr.WtW[i,j];
    }
  }

}

void drmu_basis(tree& t, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen)
{

  bool debug = false;
  tree::npv bnv;
  std::vector<sinfo> sv; //will be resized in allsuff
  tree::npv bnv0; std::vector<sinfo> sv0;
  //debug is broken, would need to eat node_pointers. suff stat code works anyhow
//  if(debug) allsuff_basis(t,xi,di,bnv0,sv0,node_pointers);

  bnv.clear();      // Clear the bnv variable if any value is already saved there.
  t.getbots(bnv);   // Save bottom nodes for x to bnv variable.
  tree::npv::size_type nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.

  double prior_prec = pi.Prec0(0,0);
  double prior_mean = pi.mu0(0);

  for(tree::npv::size_type i=0; i<nb; i++) {
    sv[i] = bnv[i]->s;
    if(debug) {
      if(abs(sv[i].sy - sv0[i].sy)>1e-8) {

        Rcout << "depth "<< bnv[i]->depth() << endl;
        Rcout << " good " << sv0[i].sy << " " << sv0[i].n << " " << sv0[i].sy_vec << sv0[i].WtW << endl;
        Rcout << " new " << sv[i].sy << " " << sv[i].n << " " << sv[i].sy_vec << sv[i].WtW << endl;
        stop("shit");
      }
     else {
        //Rcout << "depth "<< bnv[i]->depth() << " good " << sv0[i].sy << " " << sv0[i].n << " " << sv0[i].sy_vec <<" new " << sv[i].sy << " " << sv[i].n << " " << sv[i].sy_vec << endl;
        //Rcoutt << " good " << sv0[i].sy << " " << sv0[i].n << " " << sv0[i].sy_vec << sv0[i].WtW << endl;
       //Rcoutt << " new " << sv[i].sy << " " << sv[i].n << " " << sv[i].sy_vec << sv[i].WtW << endl;
      }
     sv[i] = sv0[i];
    }

  }

  if(di.basis_dim<2) {
    vec beta_draw(1);
    double tt = prior_prec*prior_mean;
    double s2 = (pi.sigma*pi.sigma);
    for(tree::npv::size_type i=0;i!=bnv.size();i++) {

      double Phi = sv[i].n0/s2 + prior_prec;
      double m = tt + sv[i].sy/s2;
      beta_draw(0) = m/Phi + gen.normal(0,1)/sqrt(Phi); //rmvnorm_post(m, Phi);

      // Assign botton node values to new mu draw.
      bnv[i] -> setm(beta_draw);

      // Check for NA result.
      if(beta_draw(0) != beta_draw(0)) {
        Rcpp::stop("drmu failed");
      }
    }
  } else {
    mat Phi;
    vec m;
    vec beta_draw;

    vec tt = pi.Prec0*pi.mu0;
    double s2 = (pi.sigma*pi.sigma);
    for(tree::npv::size_type i=0;i!=bnv.size();i++) {

      //Rcoutt << "phi ";
      //Rcoutt << "WtW" << endl << sv[i].WtW << endl;
      Phi = sv[i].WtW/s2 + pi.Prec0;
      //Rcoutt << "m ";
      m = tt + sv[i].sy_vec/s2;
      //Rcoutt << "draw ";
      beta_draw = rmvnorm_post(m, Phi);

      // Assign botton node values to new mu draw.
      bnv[i] -> setm(beta_draw);

      // Check for NA result.
      if(sum(bnv[i]->getm() == bnv[i]->getm()) == 0) {
        Rcpp::stop("drmu failed");
      }
    }
  }

}

void drmu_sumtozero(tree& t, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen)
{
  
  bool debug = false;
  tree::npv bnv;
  std::vector<sinfo> sv; //will be resized in allsuff
  tree::npv bnv0; std::vector<sinfo> sv0;
  //debug is broken, would need to eat node_pointers. suff stat code works anyhow
  //  if(debug) allsuff_basis(t,xi,di,bnv0,sv0,node_pointers);
  
  bnv.clear();      // Clear the bnv variable if any value is already saved there.
  t.getbots(bnv);   // Save bottom nodes for x to bnv variable.
  tree::npv::size_type nb = bnv.size();   // Initialize new var nb of type bvsz for number of bottom nodes, then...
  sv.resize(nb);          // Re-sizing suff stat vector to have same size as bottom nodes.
  
  double prior_prec = pi.Prec0(0,0);
  double prior_mean = pi.mu0(0);
  
  for(tree::npv::size_type i=0; i<nb; i++) {
    sv[i] = bnv[i]->s;
    if(debug) {
      if(abs(sv[i].sy - sv0[i].sy)>1e-8) {
        
        Rcout << "depth "<< bnv[i]->depth() << endl;
        Rcout << " good " << sv0[i].sy << " " << sv0[i].n << " " << sv0[i].sy_vec << sv0[i].WtW << endl;
        Rcout << " new " << sv[i].sy << " " << sv[i].n << " " << sv[i].sy_vec << sv[i].WtW << endl;
        stop("shit");
      }
      else {
        //Rcout << "depth "<< bnv[i]->depth() << " good " << sv0[i].sy << " " << sv0[i].n << " " << sv0[i].sy_vec <<" new " << sv[i].sy << " " << sv[i].n << " " << sv[i].sy_vec << endl;
        //Rcoutt << " good " << sv0[i].sy << " " << sv0[i].n << " " << sv0[i].sy_vec << sv0[i].WtW << endl;
        //Rcoutt << " new " << sv[i].sy << " " << sv[i].n << " " << sv[i].sy_vec << sv[i].WtW << endl;
      }
      sv[i] = sv0[i];
    }
  }
  
  if(di.basis_dim<2) {
    
    vec m; mat Phi, Pw;
    
    size_t nb = sv.size();
    arma::colvec sy(nb);
    arma::colvec nleaf(nb);
    arma::colvec n_by_basis(nb);
    
    for(size_t i=0; i<sv.size(); ++i) {
      sy[i] = sv[i].sy_vec(0);
      nleaf[i] = sv[i].n;
      n_by_basis[i] = sv[i].WtW(0,0);
    }
    
    s2z_stats(m, Phi, Pw, sy, n_by_basis, nleaf, pi);
    
    //Rcpp::Rcout << endl << " draw" << endl;
    arma::vec mustar = rmvnorm_post(m, Phi);
    
    //Rcpp::Rcout <<" project" << endl;
    arma::vec mu = Pw*mustar;
    vec beta_draw(1);
    for(tree::npv::size_type i=0;i!=bnv.size();i++) {
      
      beta_draw(0) = mu(i); //rmvnorm_post(m, Phi);
      
      // Assign botton node values to new mu draw.
      bnv[i] -> setm(beta_draw);
      
      // Check for NA result.
      if(beta_draw(0) != beta_draw(0)) {
        Rcpp::stop("drmu failed");
      }
    }
    
      
  } else {
      Rcpp::stop("stz trees require basis_dim==1");
  }
  
}

double lil_basis(sinfo& s, pinfo& pi){

//  Rcout << "log likelihood calc" << endl;

  double ll=0; double tt;

  double s2 = pi.sigma*pi.sigma;
  
  if(s.n < 1) return 0.0;

  if(false) {//(s.sy_vec.n_elem == 1) {
    /*

    This is slow as hell??
     Instead: check for 1d when computing summary stats, avoid matrix ops** everywhere**

    double precpred = pi.Prec0.at(0,0) + s.n0/s2;

    //double dotp = dot(s.sy_vec/s2, solve(precpred, s.sy_vec/s2));
    double dotp = precpred*s.sy/(s2*s2);
    double tt = 0;

    tt = -0.5*log(precpred) - 0.5*pi.logdetSigma0;

    ll = -0.5*n*log(s2) + tt + 0.5*dotp;
    */


  } else {
    mat precpred = pi.Prec0 + s.WtW/s2;

    double dotp = dot(s.sy_vec/s2, solve(precpred, s.sy_vec/s2));
    double tt = 0;

    tt = -0.5*log(det(precpred)) - 0.5*pi.logdetSigma0;

    ll = -0.5*s.n*log(s2) + tt + 0.5*dotp;
  }

  ////Rcpp::Rcout<< ll << endl;

  return(ll);
}


// computes important summary stats
void s2z_stats(arma::vec &m, //FC of mu is Phi^(-1)m
                     arma::mat &Phi, //prec mat for FC of mu
                     arma::mat &Pw, //Projection matrix to make leave pars sum to zero
                     arma::colvec &sy, 
                     arma::colvec &n_by_basis, 
                     arma::colvec &nleaf, 
                     pinfo &pi) {
  
  size_t nb = nleaf.size();
  double sigma2 = pi.sigma*pi.sigma;
  //Rcpp::Rcout << "nb " << nb << endl;
  //Rcpp::Rcout << "a ";
  arma::mat diag_n = diagmat(n_by_basis);
  Pw = arma::eye(nb,nb) - nleaf*nleaf.t()/(dot(nleaf, nleaf));
  if(nb==1) Pw = 0*Pw; //for numerics
  
  //Rcpp::Rcout << "b ";
  //Rcpp::Rcout  << "Pw " << Pw << endl;
  //Rcpp::Rcout  << "diag_n " << diag_n << endl;
  arma::mat AQQA = Pw*diag_n*Pw;
  //Rcpp::Rcout << "c ";
  Phi = arma::eye(nb,nb)*pi.Prec0[0,0] + AQQA/(sigma2);
  //Rcpp::Rcout << "d ";
  m = Pw*sy/sigma2;
  //Rcpp::Rcout << "e ";
  
}

// Returns important summary stats + lil evaluation
double s2z_stats_lil(arma::vec &m, //FC of mu is Phi^(-1)m
               arma::mat &Phi, //prec mat for FC of mu
               arma::mat &Pw, //Projection matrix to make leave pars sum to zero
               arma::colvec &sy, 
               arma::colvec &n_by_basis, 
               arma::colvec &nleaf, 
               pinfo &pi) {
  
  s2z_stats(m, //FC of mu is Phi^(-1)m
            Phi, //prec mat for FC of mu
            Pw, //Projection matrix to make leave pars sum to zero
            sy, 
            n_by_basis, 
            nleaf, 
            pi);
  
  size_t nb = nleaf.size();
  double sigma2 = pi.sigma*pi.sigma;
  double leaf_var = 1/(pi.Prec0[0,0]);
  
  double out = -0.5*nb*log(leaf_var) - 0.5*log(det(Phi))+ 0.5*dot(m, solve(Phi, m));
  
  return out;
}

double lil_basis_sumtozero(//tree& x, tree::npv& bnv,
                         std::vector<sinfo>& sv,
                         pinfo& pi
                         /*
                         arma::vec &m, //FC of mu is Phi^(-1)m
                         arma::mat &Phi, //prec mat for FC of mu
                         arma::mat &Pw, //Projection matrix to make leaf pars sum to zero
                          */
                         )
{

  size_t nb = sv.size();
  arma::colvec sy(nb);
  arma::colvec nleaf(nb);
  arma::colvec n_by_basis(nb);
  
  for(size_t i=0; i<sv.size(); ++i) {
    sy[i] = sv[i].sy_vec(0);
    n_by_basis[i] = sv[i].WtW(0,0);
    nleaf[i]= sv[i].n;
  }
  
  arma::vec m; arma::mat Phi, Pw;
  double out = s2z_stats_lil(m, Phi, Pw, sy, n_by_basis, nleaf, pi);
  
  return out;
  
}

double lil_basis(sinfo& sl, sinfo& sr, pinfo& pi){

  sinfo st = sl;
  st.n += sr.n;
  st.n0 += sr.n0;
  st.sy += sr.sy;
  st.sy_vec += sr.sy_vec;
  st.WtW += sr.WtW;
  
  if(st.n<1) return 0.0;

  double ll = lil_basis(st, pi);

  ////Rcpp::Rcout<< ll << endl;

  return(ll);
}

void fit_basis(tree& t, xinfo& xi, dinfo& di, double* fv, std::vector<tree::tree_cp>& node_pointers, bool populate, bool vanilla)
{
  double *xx;
  double *omega_i_tmp;
  tree::tree_cp bn;

  for(size_t i=0;i<di.n;i++) {
    xx = di.x + i*di.p;
    if(populate) {
      bn = t.bn(xx,xi);
      node_pointers[i] = bn;
    } else {
      bn = node_pointers[i];
    }

    omega_i_tmp = di.omega + i*di.basis_dim;
    //arma::vec omega_i(omega_i_tmp, di.basis_dim, false, false);
    //fv[i] = arma::dot(bn->getm(), omega_i);

    //ok
    if(vanilla) {
      fv[i] = bn->mu(0);
    } else {
      fv[i] = 0.0;
      for(size_t j=0; j<di.basis_dim; ++j) {
        fv[i] += omega_i_tmp[j]*bn->mu(j);
      }
    }


  }
}

double fit_i_basis(size_t& i, std::vector<tree>& t, xinfo& xi, dinfo& di, bool vanilla)
{
  double *xx;
  double fv = 0.0;
  double *omega_i_tmp;
  tree::tree_cp bn;

  for(size_t j=0;j<t.size();j++) {
    xx = di.x + i*di.p;
    bn = t[j].bn(xx,xi); //instead of dropping, using the node_pointer object

    omega_i_tmp = di.omega + i*di.basis_dim;

    //ok
    if(vanilla) {
      fv += bn->mu(0);
    } else {
      for(size_t k=0; k<di.basis_dim; ++k) {
        fv += omega_i_tmp[k]*bn->mu(k);
      }
    }

  }

  return fv;
}


//--------------------------------------------------
// normal density N(x, mean, variance)
double pn(double x, double m, double v)
{
	double dif = x-m;
	return exp(-.5*dif*dif/v)/sqrt(2*M_PI*v);
}

//--------------------------------------------------
// draw from discrete distributin given by p, return index
int rdisc(double *p, RNG& gen)
{

	double sum;
	double u = gen.uniform();

    int i=0;
    sum=p[0];
    while(sum<u) {
		i += 1;
		sum += p[i];
    }
    return i;
}

//--------------------------------------------------
//evalute tree tr on grid given by xi and write to os
void grm(tree& tr, xinfo& xi, std::ostream& os)
{
	size_t p = xi.size();
	if(p!=2) {
		Rcout << "error in grm, p !=2\n";
		return;
	}
	size_t n1 = xi[0].size();
	size_t n2 = xi[1].size();
	tree::tree_cp bp; //pointer to bottom node
	double *x = new double[2];
	for(size_t i=0;i!=n1;i++) {
		for(size_t j=0;j!=n2;j++) {
			x[0] = xi[0][i];
			x[1] = xi[1][j];
			bp = tr.bn(x,xi);
			os << x[0] << " " << x[1] << " " << bp->getm() << " " << bp->nid() << endl;
		}
	}
	delete[] x;
}

//--------------------------------------------------
//does this bottom node n have any variables it can split on.
bool cansplit(tree::tree_p n, xinfo& xi)
{
	int L,U;
	bool v_found = false; //have you found a variable you can split on
	size_t v=0;
	while(!v_found && (v < xi.size())) { //invar: splitvar not found, vars left
		L=0; U = xi[v].size()-1;
		n->rg(v,&L,&U);
		if(U>=L) v_found=true;
		v++;
	}
	return v_found;
}

//--------------------------------------------------
//compute prob of a birth, goodbots will contain all the good bottom nodes
double getpb(tree& t, xinfo& xi, pinfo& pi, tree::npv& goodbots)
{
	double pb;  //prob of birth to be returned
	tree::npv bnv; //all the bottom nodes
	t.getbots(bnv);
	for(size_t i=0;i!=bnv.size();i++)
		if(cansplit(bnv[i],xi)) goodbots.push_back(bnv[i]);
	if(goodbots.size()==0) { //are there any bottom nodes you can split on?
		pb=0.0;
	} else {
		if(t.treesize()==1) pb=1.0; //is there just one node?
		else pb=pi.pb;
	}
	return pb;
}

//--------------------------------------------------
//find variables n can split on, put their indices in goodvars
void getgoodvars(tree::tree_p n, xinfo& xi,  std::vector<size_t>& goodvars)
{
	int L,U;
	for(size_t v=0;v!=xi.size();v++) {//try each variable
		L=0; U = xi[v].size()-1;
		n->rg(v,&L,&U);
		if(U>=L) goodvars.push_back(v);
	}
}

//--------------------------------------------------
//get prob a node grows, 0 if no good vars, else alpha/(1+d)^beta
double pgrow(tree::tree_p n, xinfo& xi, pinfo& pi)
{
	if(cansplit(n,xi)) {
		return pi.alpha/pow(1.0+n->depth(),pi.beta);
	} else {
		return 0.0;
	}
}

//--------------------------------------------------

//get counts for all bottom nodes
std::vector<int> counts(tree& x, xinfo& xi, dinfo& di, tree::npv& bnv)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
	size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x
	double y;          //current y

	bnv.clear();
	x.getbots(bnv);

	typedef tree::npv::size_type bvsz;
//	bvsz nb = bnv.size();

  std::vector<int> cts(bnv.size(), 0);

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz i=0;i!=bnv.size();i++) bnmap[bnv[i]]=i;

	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		y=di.y[i];

		tbn = x.bn(xx,xi);
		ni = bnmap[tbn];

    cts[ni] += 1;
	}
  return(cts);
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo& di,
                   tree::npv& bnv, //vector of pointers to bottom nodes
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
	double *xx;        //current x
	double y;          //current y

	typedef tree::npv::size_type bvsz;
//	bvsz nb = bnv.size();

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index

	xx = di.x + i*di.p;
	y=di.y[i];

	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];

  cts[ni] += sign;
}

void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign)
{
  tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
	double y;          //current y
  /*
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
	*/
	xx = di.x + i*di.p;
	y=di.y[i];

	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];

  cts[ni] += sign;
}


void update_counts(int i, std::vector<int>& cts, tree& x, xinfo& xi,
                   dinfo& di,
                   std::map<tree::tree_cp,size_t>& bnmap,
                   int sign,
                   tree::tree_cp &tbn
                   )
{
  //tree::tree_cp tbn; //the pointer to the bottom node for the current observations
  size_t ni;         //the  index into vector of the current bottom node
  double *xx;        //current x
  double y;          //current y
  /*
	typedef tree::npv::size_type bvsz;
	bvsz nb = bnv.size();

	std::map<tree::tree_cp,size_t> bnmap;
	for(bvsz ii=0;ii!=bnv.size();ii++) bnmap[bnv[ii]]=ii; // bnmap[pointer] gives linear index
	*/
	xx = di.x + i*di.p;
	y=di.y[i];

	tbn = x.bn(xx,xi);
	ni = bnmap[tbn];

  cts[ni] += sign;
}

bool min_leaf(int minct, std::vector<tree>& t, xinfo& xi, dinfo& di) {
  bool good = true;
  tree::npv bnv;
  std::vector<int> cts;
  int m = 0;
  for (size_t tt=0; tt<t.size(); ++tt) {
    cts = counts(t[tt], xi, di, bnv);
    m = std::min(m, *std::min_element(cts.begin(), cts.end()));
    if(m<minct) {
      good = false;
      break;
    }
  }
  return good;
}

mat coef_basis(tree& t, xinfo& xi, dinfo& di)
{
  double *xx;
  
  tree::tree_cp bn;
  
  mat out(di.basis_dim,di.n);
  
  for(size_t i=0; i<di.n; i++) {
    xx = di.x + i*di.p;
    bn = t.bn(xx,xi);
    out.col(i) = bn->getm();
  }
  return(out);
}

arma::mat coef_basis_i(size_t i, tree& t, xinfo& xi, dinfo& di)
{
  double *xx;
  
  tree::tree_cp bn;
  
  arma::mat out(di.basis_dim, 1);
  
  //for(size_t i=0; i<1; i++) {
    xx = di.x + i*di.p;
    bn = t.bn(xx,xi);
    out.col(0) = bn->getm();
  //}
  return(out);
} 


//--------------------------------------------------
//partition
void partition(tree& t, xinfo& xi, dinfo& di, std::vector<size_t>& pv)
{
	double *xx;
	tree::tree_cp bn;
	pv.resize(di.n);
	for(size_t i=0;i<di.n;i++) {
		xx = di.x + i*di.p;
		bn = t.bn(xx,xi);
		pv[i] = bn->nid();
	}
}

//--------------------------------------------------
//write cutpoint information to screen
void prxi(xinfo& xi)
{
	Rcout << "xinfo: \n";
	for(size_t v=0;v!=xi.size();v++) {
		Rcout << "v: " << v << endl;
		for(size_t j=0;j!=xi[v].size();j++) Rcout << "j,xi[v][j]: " << j << ", " << xi[v][j] << endl;
	}
	Rcout << "\n\n";
}

//--------------------------------------------------
//make xinfo = cutpoints
void makexinfo(size_t p, size_t n, double *x, xinfo& xi, size_t nc)
{
	double xinc;

	//compute min and max for each x
	std::vector<double> minx(p,INFINITY);
	std::vector<double> maxx(p,-INFINITY);
	double xx;
	for(size_t i=0;i<p;i++) {
		for(size_t j=0;j<n;j++) {
			xx = *(x+p*j+i);
			if(xx < minx[i]) minx[i]=xx;
			if(xx > maxx[i]) maxx[i]=xx;
		}
	}
	//make grid of nc cutpoints between min and max for each x.
	xi.resize(p);
	for(size_t i=0;i<p;i++) {
		xinc = (maxx[i]-minx[i])/(nc+1.0);
		xi[i].resize(nc);
		for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
	}
}
// get min/max needed to make cutpoints
void makeminmax(size_t p, size_t n, double *x, std::vector<double> &minx, std::vector<double> &maxx)
{
	double xx;

	for(size_t i=0;i<p;i++) {
		for(size_t j=0;j<n;j++) {
			xx = *(x+p*j+i);
			if(xx < minx[i]) minx[i]=xx;
			if(xx > maxx[i]) maxx[i]=xx;
		}
	}
}
//make xinfo = cutpoints give the minx and maxx vectors
void makexinfominmax(size_t p, xinfo& xi, size_t nc, std::vector<double> &minx, std::vector<double> &maxx)
{
	double xinc;
	//make grid of nc cutpoints between min and max for each x.
	xi.resize(p);
	for(size_t i=0;i<p;i++) {
		xinc = (maxx[i]-minx[i])/(nc+1.0);
		xi[i].resize(nc);
		for(size_t j=0;j<nc;j++) xi[i][j] = minx[i] + (j+1)*xinc;
	}
}

// funs for updating ordinal cutpoints: ----------
//

// Compute MVT density
double dmvt_cpp(arma::vec x,
                arma::mat Sigma, 
                arma::vec delta, 
                int df,
                bool log_bool){ // Evaluate MVT density at x (log or not)
  int p = Sigma.n_cols;
  double d1 = tgamma((df + p)/2.0) / tgamma(df/2.0) / pow(df, p/2.0) / 
    pow(arma::datum::pi, p / 2.0) / sqrt(det(Sigma));
  arma::mat m1 = arma::conv_to<arma::mat>::from(x - delta);
  arma::mat m2 = trans(m1) * inv(Sigma) * m1;
  double d2 = pow(1.0 + arma::conv_to<double>::from(m2) / df, -(df + p) / 2.0);
  if(log_bool == true){
    return(log(d1 * d2));
  } else {
    return(d1 * d2);
  }  
}

// compute hessian of joint likelihood
arma::mat hessian_cuts(arma::vec y, 
                       arma::vec mu_vect, 
                       arma::vec cuts) { 
  int J = cuts.size();
  int N = y.size();
  arma::mat hessian(J - 2, J - 2);
  for(int k = 2; k <= (J - 1); k++){
    double diag_entry = 0.0;
    for(int i = 0; i < N; i++){
      if(y[i] == k){
        double v1 = cuts(k - 1) - mu_vect(i);
        double v2 = cuts(k - 2) - mu_vect(i);
        double v3 = R::pnorm(v1, 0.0, 1.0, true, false) - R::pnorm(v2, 0.0, 1.0, true, false);
        double comp = (- v3 * R::dnorm(v1, 0.0, 1.0, false) * v1 - 
                       pow(R::dnorm(v1, 0.0, 1.0, false), 2.0)) / pow(v3, 2.0);
        diag_entry = diag_entry + comp;
      }
      if(y[i] == k + 1){
        double v1 = cuts(k) - mu_vect(i);
        double v2 = cuts(k - 1) - mu_vect(i);
        double v3 = R::pnorm(v1, 0.0, 1.0, true, false) - R::pnorm(v2, 0.0, 1.0, true, false);
        double comp = (v3 * R::dnorm(v2, 0.0, 1.0, false) * v2 - 
                       pow(R::dnorm(v2, 0.0, 1.0, false), 2.0)) / pow(v3, 2.0);
        diag_entry = diag_entry + comp;
      }
    }
    hessian(k-2, k-2) = diag_entry;
  }
  for(int k = 3; k <= (J - 1); k++){
    double diag_entry = 0.0;
    for(int i = 0; i < N; i++){
      if(y[i] == k){
        double v1 = cuts(k - 1) - mu_vect(i);
        double v2 = cuts(k - 2) - mu_vect(i);
        double comp = R::dnorm(v1, 0.0, 1.0, false) * R::dnorm(v2, 0.0, 1.0, false) /
          pow(R::pnorm(v1, 0.0, 1.0, true, false) - R::pnorm(v2, 0.0, 1.0, true, false), 2);
        diag_entry = diag_entry + comp;
      }
    }
    hessian(k-3, k-2) = diag_entry;
  }
  for(int k = 2; k <= (J - 2); k++){
    double diag_entry = 0.0;
    for(int i = 0; i < N; i++){
      if(y[i] == k + 1){
        double v1 = cuts(k) - mu_vect(i);
        double v2 = cuts(k - 1) - mu_vect(i);
        double comp = R::dnorm(v1, 0.0, 1.0, false) * R::dnorm(v2, 0.0, 1.0, false) / 
          pow(R::pnorm(v1, 0.0, 1.0, true, false) - R::pnorm(v2, 0.0, 1.0, true, false), 2);
        diag_entry = diag_entry + comp;
      }
    }
    hessian(k-1, k-2) = diag_entry;
  }
  
  return hessian;
}

// compute gradient of joint likelihood
arma::rowvec gradient_cuts(arma::vec y, 
                           arma::vec mu_vect, 
                           arma::vec cuts) { 
  int J = cuts.size();
  int N = y.size();
  arma::rowvec gradient(J - 2);
  for(int k = 2; k <= (J - 1); k++){
    double grad_entry = 0;
    for(int i = 0; i < N; i++){
      if(y[i] == k){
        double v1 = cuts(k - 1) - mu_vect(i);
        double v2 = cuts(k - 2) - mu_vect(i);
        grad_entry = grad_entry + R::dnorm(v1, 0.0, 1.0, false) /
          (R::pnorm(v1, 0.0, 1.0, true, false) - R::pnorm(v2, 0.0, 1.0, true, false));
      }
      
      if(y[i] == k + 1){
        double v1 = cuts(k) - mu_vect(i);
        double v2 = cuts(k - 1) - mu_vect(i);
        grad_entry = grad_entry - R::dnorm(v2, 0.0, 1.0, false) /
          (R::pnorm(v1, 0.0, 1.0, true, false) - R::pnorm(v2, 0.0, 1.0, true, false));
      }
    }
    gradient(k - 2) = grad_entry;
  }
  return gradient;
}

// compute log posterior of joint likelihood
// [[Rcpp::export]]
double log_post_cuts (arma::vec cuts_free,
                      arma::vec y, 
                      arma::vec mu_vect) { 
  int J = cuts_free.size() + 2;
  int N = y.size();
  double lp = 0;
  for(int i = 0; i < N; i++){
    if(y(i) == 2){
      double a = log(R::pnorm(cuts_free(0) - mu_vect(i), 0.0, 1.0, true, false) -
                     R::pnorm(0 - mu_vect(i), 0.0, 1.0, true, false));
      lp = lp + a;
    }
    
    for(int j = 3; j <= (J - 1); j++){
      if(y(i) == j){
        double a = log(R::pnorm(cuts_free(j - 2) - mu_vect(i), 0.0, 1.0, true, false) -
                       R::pnorm(cuts_free(j - 3) - mu_vect(i), 0.0, 1.0, true, false));
        lp = lp + a;
      }
    }
    
    if(y(i) == J){
      double a = log(1 - R::pnorm(cuts_free(J - 3) - mu_vect(i), 0.0, 1.0, true, false));
      lp = lp + a;
    }
  }
  return(lp);
}

// Newton Raphson optimization
arma::vec Newt_Raph_cuts(arma::vec y, 
                         arma::vec mu_vect, 
                         arma::vec cuts) {
  int J = cuts.size();
  arma::vec cutpoints_temp = cuts;
  int B = 15;
  for(int i = 0; i <= B; i++){
    arma::mat inv_hess = arma::inv(hessian_cuts(y, mu_vect, cutpoints_temp));
    arma::rowvec grad = gradient_cuts(y, mu_vect, cutpoints_temp);
    arma::mat diff = inv_hess * trans(arma::conv_to<arma::mat>::from(grad));
    for(int j = 0; j <= (J - 3); j ++){
      cutpoints_temp(j + 1) = cutpoints_temp(j + 1) - diff(j);
    }
  }
  return(cutpoints_temp);
}


// Update Omega
arma::mat Omega_update(arma::vec x,
                       int j = 12,
                       double L = 2.5,
                       double lscale = 1/(M_PI*2),
                       int sigma = 1) {
  size_t N = x.size();
  arma::mat Omega(N, j);
  for(int i = 0; i < j; i++){
    for(size_t k = 0; k < N; k++){
      Omega(k,i) = sqrt(1 / L) * sin(M_PI_2 * (i + 1) * (x(k) + L) / L);
    }
  }
  //Rcout << lscale << endl;
  
  arma::vec V(j);
  for(int i = 0; i < j; i++){
    double w = M_PI * (i + 1) / (2 * L);
    double s = pow(sigma, 2) * sqrt(2 * M_PI) * lscale * exp(- 0.5 * pow(lscale, 2) * pow(w,2));
    V(i) = s; 
  }
  arma::mat Vmat = sqrt(arma::diagmat(V));
  
  return(Omega * Vmat.t());
}

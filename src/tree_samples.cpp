#include "tree_samples.h"
#include <RcppArmadillo.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>

#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"

using namespace Rcpp;

void tree_samples::load_string(CharacterVector samples_) {
  Rcout << "Loading...\n";
//    std::string samples_str = as<std::string>(samples_); 
  std::stringstream treef;
  treef.str(as<std::string>(samples_));
  treef >> xi; //load the cutpoints
  treef >> m;  //number of trees
  Rcout << "ntrees " << m << endl;
  treef >> p;  //dimension of x's
  Rcout << "p " << p << endl;
  treef >> basis_dim;  //dimension of basis functions
  Rcout << "leaf parameter dimension " << basis_dim << endl;
  treef >> ndraws; //number of draws from the posterior that were saved.
  Rcout << "ndraws " << ndraws << endl;
  
  t.resize(ndraws,std::vector<tree>(m));
  for(size_t i=0;i<ndraws;i++) {
    for(size_t j=0;j<m;j++) {
      treef >> t[i][j];
    }
  }
  Rcout << "done" <<endl;
  init = true;
}

CharacterVector tree_samples::save_string() {
  Rcout << "Saving state...\n";
  //    std::string samples_str = as<std::string>(samples_); 
  std::stringstream treef;
  treef << xi << endl; //load the cutpoints
  treef << m << endl;  //number of trees
  Rcout << "ntrees " << m << endl;
  treef << p << endl;  //dimension of x's
  Rcout << "p " << p << endl;
  treef << basis_dim << endl;  //dimension of basis functions
  Rcout << "leaf parameter dimension " << basis_dim << endl;
  treef << ndraws << endl; //number of draws from the posterior that were saved.
  Rcout << "ndraws " << ndraws << endl;
  
  for(size_t i=0;i<ndraws;i++) {
    for(size_t j=0;j<m;j++) {
      treef << t[i][j];
    }
  }
  Rcout << "done" <<endl;
  
  Rcpp::CharacterVector ret;
  ret = treef.str();
  
  return(ret);
}

//deprecated
void tree_samples::load_list(List samples_, List x_info_list) {
  Rcout << "Loading...\n";
  //    std::string samples_str = as<std::string>(samples_); 

  p = x_info_list.size();
  xi.resize(p);
  for(int j=0; j<p; ++j) {
    NumericVector tmp = x_info_list[j];
    std::vector<double> tmp2;
    for(size_t s=0; s<tmp.size(); ++s) {
      tmp2.push_back(tmp[s]);
    }
    xi[j] = tmp2;
  }
  
//    treef >> xi; //load the cutpoints -- TODO
  List first_draw = samples_[0];
  m = first_draw.size();  //number of trees
  Rcout << "ntrees " << m << endl;
  //treef >> p;  //dimension of x's
  Rcout << "p " << p << endl;
  List first_tree = first_draw[0];
  List first_node = first_tree[first_tree.size()-1];
  vec first_par = first_node["m"];
  basis_dim = first_par.size();  //dimension of basis functions
  Rcout << "leaf parameter dimension " << basis_dim << endl;
  
  ndraws = samples_.size();
  Rcout << "ndraws " << ndraws << endl;
  
  
  t.resize(ndraws,std::vector<tree>(m));
  for(size_t i=0;i<ndraws;i++) {
    List flat_trees = samples_[i]; // List of trss from draw i
    for(size_t j=0;j<m;j++) {
      //Rcout << "i " << i << " j " << j << endl;
      List ft = flat_trees[j]; // tree j from draw i
      unflatten(ft, t[i][j]);
    }
  }
  
  Rcout << "done" <<endl;
  init = true;
}

void tree_samples::scale(double s) { //BROKEN from R?

  for(size_t i=0;i<ndraws;i++) {
    for(size_t j=0; j<m; ++j) {
      //  		    Rcout << t[i][j] << endl << endl;
      t[i][j].scale(s);
    }
  }

}


//uint32_t u = reinterpret_cast<uintptr_t>(node_pointers[s][j][ii]);

// Returns a vector of ints. Each entry in the vector is a unique sequential leaf id
// across all trees except the last entry, which gives the max id
std::vector<imat> tree_samples::dummy_sparse(NumericMatrix &x_) {
  size_t n = x_.ncol();
  Rcout << "Predicting for " << n << " new observations" << endl;
  std::vector<imat> ret(ndraws);
  if(init) {
    
    dinfo di;
    di.n=n; di.p=p; di.x = &x_[0]; di.y=0; di.basis_dim = basis_dim;
    
    double *xx;
    tree::tree_cp bn;
    tree::npv bnv;
    typedef tree::npv::size_type bvsz;
    std::map<tree::tree_cp,size_t> bnmap;
    
    for(size_t i=0;i<ndraws;i++) {
      
      //build node - column map
      bnmap.clear();
      size_t colix = 0;
      for(size_t j=0; j<m; ++j) {
        bnv.clear();
        t[i][j].getbots(bnv);
        //Rcout << bnv.size() << endl;
        for(bvsz q=0;q!=bnv.size();q++) { 
          bnmap[bnv[q]]=colix;
          colix++;
          //Rcout << colix << " ";
        }
      }
      
      ret[i].resize(m*n+1, 1);
      
      for(size_t j=0; j<m; ++j) {
        for(size_t k=0; k<n; ++k){
          xx = di.x + k*di.p;
          bn = t[i][j].bn(xx,xi);
          ret[i](n*j+k, 0) = bnmap[bn];
        }
      }
      
      ret[i][m*n] = colix; //last entry gives the column index
      
    }
    //      }
  } else {
    Rcout << "Uninitialized" <<'\n';
  }
  return ret;
  
}

std::vector<std::vector<double> > tree_samples::mu_vec(NumericMatrix &x_) {
  size_t n = x_.ncol();
  Rcout << "Predicting for " << n << " new observations" << endl;
  std::vector<std::vector<double> > ret(ndraws);
  if(init) {
    
    //std::vector<double> x;
    //for(NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) x.push_back(*it);
    
    dinfo di;
    di.n=n; di.p=p; di.x = &x_[0]; di.y=0; di.basis_dim = basis_dim;
    
    double *xx;
    tree::tree_cp bn;
    tree::npv bnv;
    typedef tree::npv::size_type bvsz;
    std::map<tree::tree_cp,size_t> bnmap;
    
    for(size_t i=0;i<ndraws;i++) {
      
      std::vector<double> mu;
      //build node - column map
      //bnmap.clear();
      //size_t colix = 0;
      for(size_t j=0; j<m; ++j) {
        bnv.clear();
        t[i][j].getbots(bnv);
        //Rcout << bnv.size() << endl;
        for(bvsz q=0;q!=bnv.size();q++) { 
          arma::vec t = bnv[q]->getm();
          double m = t(0);
          mu.push_back(m);
        }
        ret[i] = mu;
      }
    }
  } else {
    Rcout << "Uninitialized" <<'\n';
  }
  return ret;
  
}

std::vector<std::vector<std::vector<size_t > > >  tree_samples::dummies(NumericMatrix &x_) {
  size_t n = x_.ncol();
  Rcout << "Predicting for " << n << " new observations" << endl;
  std::vector<std::vector<std::vector<size_t > > >  ret(ndraws);
  if(init) {
    for(size_t i=0;i<ndraws;i++) {
      ret[i].resize(m);
      for(size_t j=0; j<m; ++j) {
        ret[i][j].resize(n);
      }
    }
    
    std::vector<double> x;
    for(NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) x.push_back(*it);
    
    dinfo di;
    di.n=n; di.p=p; di.x = &x[0]; di.y=0; di.basis_dim = basis_dim;
    
    double *xx;
    tree::tree_cp bn;
    
    for(size_t i=0;i<ndraws;i++) {
      for(size_t j=0; j<m; ++j) {
        for(size_t k=0; k<n; ++k){
          xx = di.x + k*di.p;
          bn = t[i][j].bn(xx,xi);
          ret[i][j][k] = bn->nid();
        }
      }
    }
    //      }
  } else {
    Rcout << "Uninitialized" <<'\n';
  }
  return ret;
}

arma::cube tree_samples::coefs(NumericMatrix &x_) {
  size_t n = x_.ncol();
  Rcout << "Predicting for " << n << " new observations" << endl;
  arma::cube ret(basis_dim, n, ndraws);
  ret.fill(0);
  if(init) {
    std::vector<double> x;
    for(NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) x.push_back(*it);
    
    dinfo di;
    di.n=n; di.p=p; di.x = &x[0]; di.y=0; di.basis_dim = basis_dim;

    for(size_t i=0;i<ndraws;i++) {
		  for(size_t j=0; j<m; ++j) {
//  		    Rcout << t[i][j] << endl << endl;
		    ret.slice(i) += coef_basis(t[i][j], xi, di);
		  }
    }
//      }
  } else {
    Rcout << "Uninitialized" <<'\n';
  }
  return ret;
}

arma::mat tree_samples::fits(NumericMatrix &x_, NumericMatrix &Omega_) {
  size_t n = x_.ncol();
  Rcout << "Predicting for " << n << " new observations" << endl;
  arma::mat ret(n, ndraws);
  ret.fill(0);
  
  arma::mat Omega = as<arma::mat>(Omega_);
  
  if(init) {
    std::vector<double> x;
    for(NumericMatrix::iterator it=x_.begin(); it!=x_.end(); ++it) x.push_back(*it);
    
    dinfo di;
    di.n=n; di.p=p; di.x = &x[0]; di.y=0; di.basis_dim = basis_dim;
    
    arma::rowvec tm(Omega.n_cols);
    
    for(size_t i=0;i<ndraws;i++) {
      for(size_t j=0; j<m; ++j) {
        tm = sum(Omega%coef_basis(t[i][j], xi, di));
        ret.col(i) += tm.t();
      }
    }
    //      }
  } else {
    Rcout << "Uninitialized" <<'\n';
  }
  return ret;
}

RCPP_MODULE(tree_samples) {
  class_<tree_samples>( "tree_samples" )
  .constructor()
  .field("ntree", &tree_samples::m, "Number of trees")
  .field("basis_dim", &tree_samples::basis_dim, "Dimension of leaf parameters")
  .method( "load_string", &tree_samples::load_string )
  .method( "save_string", &tree_samples::save_string )
  .method( "load_list", &tree_samples::load_list )
  .method( "scale", &tree_samples::load_string )
  //  .method( "load_file", &tree_samples::load_file )
      .method( "coefs", &tree_samples::coefs  )
      .method( "fits", &tree_samples::fits  )
      .method( "dummies", &tree_samples::dummies  )
      .method( "dummy_sparse", &tree_samples::dummy_sparse  )
      .method( "mu_vec", &tree_samples::mu_vec  )
  ;
} 
//   
// // [[Rcpp::export]]
// tree_samples test_ts() {
//   tree_samples out;
//   out.m = 14;
//   return(out);
// }

// 
// // invoke the draw method
// // [[Rcpp::export]]
// void test_tree_samples(SEXP pts) {
//   // grab the object as a XPtr (smart pointer)
//   // to UniformRcpp::XPtr<Uniform> ptr(xp);
//   // convert the parameter to int
//   // invoke the function
//   
//   XPtr<tree_samples> xpts(pts);
//   
//   Rcout << xpts->m;
//   // return the result to R
// 
// }
// 
// // invoke the draw method
// // [[Rcpp::export]]
// Rcpp::XPtr<tree_samples> retree(SEXP pts) {
//   // grab the object as a XPtr (smart pointer)
//   // to UniformRcpp::XPtr<Uniform> ptr(xp);
//   // convert the parameter to int
//   // invoke the function
//   
//   XPtr<tree_samples> xpts(pts);
//   
//   Rcout << xpts->m;
//   // return the result to R
//   
// }
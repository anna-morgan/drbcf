#include "arma_config.h"
#include <RcppArmadillo.h>
#include <RcppArmadilloExtensions/sample.h>
#include <cstdint>


#include <iostream>
#include <fstream>
#include <vector>
#include <ctime>
#include <cereal/archives/portable_binary.hpp>
#include <cereal/archives/binary.hpp>
#include <cereal/archives/xml.hpp>


#include "rng.h"
#include "tree.h"
#include "info.h"
#include "funs.h"
#include "bd.h"
#include "tree_samples.h"

using namespace Rcpp;


// [[Rcpp::export]]
List multibart(arma::vec y_,
                 List &bart_specs,
                 List &bart_designs,
                 arma::mat random_des,
                 arma::mat random_var, arma::mat random_var_ix, //random_var_ix*random_var = diag(Var(random effects))
                 double random_var_df, arma::vec randeff_scales,
                 int burn, int nd, int thin, //Draw nd*thin + burn samples, saving nd draws after burn-in
                 double lambda, double nu, //prior pars for sigma^2_y
                 bool return_trees = true,
                 bool save_trees = false,
                 bool est_mod_fits = false, bool est_con_fits = false,
                 bool prior_sample = false,
                 int status_interval = 100,
                 NumericVector lower_bd = NumericVector::create(0.0),
                 NumericVector upper_bd = NumericVector::create(0.0),
                 bool ordinal = false,
                 NumericVector y_obs = NumericVector::create(0.0),
                 NumericVector cutpoints_init = NumericVector::create(0.0),
                 NumericVector basis_params = NumericVector::create(0.0),
                 bool text_trace = true,
                 bool R_trace = false,
                 bool return_coefs = false)
{
  
  //Rcout << paths[0] << endl << paths[1];
  
  bool randeff = true;
  if(random_var_ix.n_elem == 1) {
    randeff = false;
  }
  
  if(randeff) Rcout << "Using random effects." << std::endl;
  
  //  std::string treef_name = as<std::string>(treef_name_);
  //  std::ofstream treef(treef_name.c_str());
  //  std::string treef_name_serial = as<std::string>(treef_name_serial_);
  //  std::ofstream treef_serial(treef_name_serial.c_str(), std::ios::out | std::ios::binary);
  
  RNGScope scope;
  RNG gen; //this one random number generator is used in all draws
  
  //Rcout << "\n*****Into bart main\n";
  
  /*****************************************************************************
   /* Read, format y
    *****************************************************************************/
   Rcout << "Reading in y\n\n";
   
   std::vector<double> y; //storage for y
   double miny = INFINITY, maxy = -INFINITY;
   sinfo allys;       //sufficient stats for all of y, use to initialize the bart trees.
   
   for(NumericVector::iterator it=y_.begin(); it!=y_.end(); ++it) {
     y.push_back(*it);
     if(*it<miny) miny=*it;
     if(*it>maxy) maxy=*it;
     allys.sy += *it; // sum of y
     allys.sy2 += (*it)*(*it); // sum of y^2
   }
   size_t n = y.size();
   allys.n = n;
   
   double ybar = allys.sy/n; //sample mean
   double shat = sqrt((allys.sy2-n*ybar*ybar)/(n-1)); //sample standard deviation
   if(ordinal) shat = 1.0;
   
   double sigma = shat;
   
   /*****************************************************************************
    /* Read, format design info
     *****************************************************************************/
    
    Rcout << "Setting up designs\n\n";
    
    size_t num_designs = bart_designs.size();
    
    std::vector<std::vector<double> > x(num_designs);
    std::vector<std::vector<int> > groups(num_designs);
    std::vector<bool> group(num_designs);
    std::vector<xinfo> x_info(num_designs);
    std::vector<arma::mat> Omega(num_designs);
    std::vector<arma::mat> Qt(num_designs);
    std::vector<arma::mat> R(num_designs);
    std::vector<size_t> covariate_dim(num_designs);
    
    //For DART
    std::vector<std::vector<int> > vargrps(num_designs);
    std::vector<int> unique_vars(num_designs);
    std::vector<std::vector<double> > varwts(num_designs);
    
    //For Omega Update:
    int nb = 0;
    double L = 0.;
    double lscale = 0.;
    double sig_omega = 1.;
    bool update_omega_bool = false;
    if(basis_params.size() > 1){
      nb = basis_params[0];
      L = basis_params[1];
      lscale = basis_params[2];
      sig_omega = 1.;
      update_omega_bool = true;
    }
    
    for(size_t i=0; i<num_designs; i++) {
      
      Rcout << "design " << i << endl;
      
      List dtemp = bart_designs[i];
      
      //Rcout << "desi " << i << endl;
      
      //the n*p numbers for x are stored as the p for first obs, then p for second, and so on.
      //std::vector<double> x_con;
      
      bool g = dtemp["group"];
      group[i] = g;
      
      //Rcout << "group setting: " << g << endl;
      
      IntegerVector gt_ = dtemp["groups"];
      for(IntegerVector::iterator it=gt_.begin(); it!= gt_.end(); ++it) {
        groups[i].push_back(*it);
      }
      
      NumericVector xt_ = dtemp["X"]; // shallow copy
      //for(NumericVector::iterator it=xt_.begin(); it!= xt_.end(); ++it) {
      //  x[i].push_back(*it);
      //}
      size_t p = xt_.size()/n;
      covariate_dim[i] = p;
      
      IntegerVector vg_ = dtemp["vargroups"];
      //Rcout << " vargroups " << vg_ << endl;
      for(IntegerVector::iterator it=vg_.begin(); it!= vg_.end(); ++it) {
        vargrps[i].push_back(*it);
      }
      
      NumericVector vw_ = dtemp["varwts"];
      Rcout << " varwts " << vw_ << endl;
      for(NumericVector::iterator it=vw_.begin(); it!= vw_.end(); ++it) {
        varwts[i].push_back(*it);
      }
      
      int uv = dtemp["unique_vars"];
      unique_vars[i] = uv;
      
      Rcout << "Instantiated covariate matrix " << i+1 << " with " << p << " columns" << endl;
        
        //Rcout << "a " << i << endl;
        
        xinfo xi;
        xi.resize(p);
        List x_info_list = dtemp["info"];
        for(int j=0; j<p; ++j) {
          NumericVector tmp = x_info_list[j];
          std::vector<double> tmp2;
          for(size_t s=0; s<tmp.size(); ++s) {
            tmp2.push_back(tmp[s]);
          }
          xi[j] = tmp2;
        }
        x_info[i] = xi;
        
        //Rcout << "b " << i << endl;
        
        Omega[i] = as<arma::mat>(dtemp["Omega"]);
        Qt[i] = as<arma::mat>(dtemp["Qt"]);
        R[i]  = as<arma::mat>(dtemp["R"]);
        
    }
    
    /*****************************************************************************
     /* Set up forests
      *****************************************************************************/
     
     Rcout << "Setting up forests\n\n";
    
    size_t num_forests = bart_specs.size();
    std::vector<std::vector<tree> > trees(num_forests);
    std::vector<pinfo> prior_info(num_forests);
    std::vector<std::vector<double> > allfits(num_forests);
    std::vector<double> r_tree(n);
    std::fill(r_tree.begin(), r_tree.end(), 0.0);
    
    std::vector<dinfo> di(num_forests);
    std::vector<std::vector<std::vector<tree::tree_cp> > > node_pointers(num_forests);
    //std::vector<std::vector<std::vector<std::vector<std::uintptr_t> > > > node_pointer_trace(nd);
    //for(size_t s=0; s<nd; ++s) node_pointer_trace[s].resize(num_forests);
    // node_pointer_trace[iteration][forest][tree][obs]
    std::vector<double> sample_eta(num_forests);
    
    std::vector<std::stringstream> tree_streams(num_forests);
    std::vector<std::stringstream> serial_streams(num_forests);
    std::vector<Rcpp::CharacterVector> Rtree_streams(num_forests);
    
    std::vector<tree_samples> final_tree_trace(num_forests);
    
    double* ftemp  = new double[n]; //fit of current tree
    for(size_t i=0; i<num_forests; ++i) {
      
      Rcout << "Forest " << i+1 << endl;
      
      //Rcout << i << endl;
      List spec = bart_specs[i];

      double ortho = spec["ortho"];
      if(ortho>0) prior_info[i].ortho = 1;
            
      double ns = spec["nosplits"];
      if(ns>0) prior_info[i].nosplits = 1;
      
      double es = spec["sample_eta"];
      sample_eta[i] = es;
      
      int desi = spec["design_index"];
      size_t ntree = spec["ntree"];
      trees[i].resize(ntree);
      prior_info[i].vanilla = spec["vanilla"];
      
      //tree_streams[i].precision(10);
      /*
       tree_streams[i] << x_info[desi] << endl;
       tree_streams[i] << ntree << endl;
       tree_streams[i] << covariate_dim[desi] << endl;
       tree_streams[i] << Omega[desi].n_rows << endl;
       tree_streams[i] << nd << endl;
       */
      
      /*
       //save stuff to tree file
       treef << xi << endl; //cutpoints
       treef << m << endl;  //number of trees
       treef << p << endl;  //dimension of x's
       treef << (int)(nd/thin) << endl;
       */
      
      //Rcout << "a" << endl;
      
      for(size_t j=0; j<ntree; ++j) trees[i][j].setm(zeros(Omega[desi].n_rows));
      
      prior_info[i].pbd = 1.0; //prob of birth/death move
      prior_info[i].pb = .5; //prob of birth given  birth/death
      
      prior_info[i].alpha = spec["alpha"]; //prior prob a bot node splits is alpha/(1+d)^beta, d is depth of node
      prior_info[i].beta  = spec["beta"];
      prior_info[i].sigma = shat;
      
      double s2z = spec["sum_to_zero"];
      if(s2z>0) prior_info[i].sum_to_zero  = true;
      
      prior_info[i].mu0 = as<arma::vec>(spec["mu0"]);
      
      //Rcout << "b" << endl;
      
      prior_info[i].Sigma0 = as<arma::mat>(spec["Sigma0"]);
      prior_info[i].Prec0 = prior_info[i].Sigma0.i();
      prior_info[i].logdetSigma0 = log(det(prior_info[i].Sigma0));
      prior_info[i].eta = 1;
      prior_info[i].gamma = 1;
      prior_info[i].scale_df = spec["scale_df"];
      
      //Rcout << "c" << endl;
      
      //Rcout << "d" << endl;
      
      //Rcout << desi << endl;
      
      //data info
      dinfo dtemp;
      dtemp.n=n;
      //Rcout << "d1" << endl;
      dtemp.p = covariate_dim[desi];
      //Rcout << "d11" << endl;
      //dtemp.x = &(x[desi])[0];
      
      //Rcout <<"aa " << endl;
      List bdd = bart_designs[desi];
      NumericVector xt_ = bdd["X"];
      //Rcout <<"bb " << endl;
      dtemp.x = &(xt_)[0];
      //Rcout <<"cc " << endl;
      
      //Rcout << "ptr test " << &(x[desi])[0] << " " <<  &x[desi][0] << endl;
      //Rcout << "ptr test " << *(&x[desi][0]) << " " <<  *(&x[desi][3]) << endl;
      //Rcout << "d12" << endl;
      dtemp.y = &r_tree[0]; //the y for each draw will be the residual
      //Rcout << "d2" << endl;
      dtemp.basis_dim = Omega[desi].n_rows;
      dtemp.omega = &(Omega[desi])[0];
      
      dtemp.cov_dim = Qt[desi].n_rows;
      dtemp.Qt = &(Qt[desi])[0];
      dtemp.R = &(R[desi])[0];
      
      //Rcout<< "groups" << endl;
      dtemp.groups = &(groups[desi])[0];
      //Rcout<< "group" << endl;
      dtemp.group = group[desi];
      
      //Rcout << "e" << endl;
      
      // Initialize node pointers & allfits
      node_pointers[i].resize(ntree);
      //for(size_t s=0; s<nd; ++s) node_pointer_trace[s][i].resize(ntree);
      allfits[i].resize(n);
      std::fill(allfits[i].begin(), allfits[i].end(), 0.0);
      for(size_t j=0; j<ntree; ++j) {
        node_pointers[i][j].resize(n);
        //for(size_t s=0; s<nd; ++s) node_pointer_trace[s][i][j].resize(n);
        fit_basis(trees[i][j],x_info[desi],dtemp,ftemp,node_pointers[i][j],true,prior_info[i].vanilla);
        //fits
        for(size_t k=0; k<n; ++k) allfits[i][k] += ftemp[k];
        
        //Rcout << "allfits test" << allfits[i][3] << " " << allfits[i][60] << endl;
      }
      
      // DART
      double dt = spec["dart"];
      if(dt>0) prior_info[i].dart = true;
      Rcout << "DART is " << prior_info[i].dart << endl;
      if(prior_info[i].dart) prior_info[i].dart_alpha = 1.0;
      std::vector<double> vp(covariate_dim[desi], 1.0/covariate_dim[desi]);
      prior_info[i].var_probs = vp;
      std::vector<double> vps(unique_vars[desi], 1.0/unique_vars[desi]);
      prior_info[i].var_probs_short = vps;
      
      //Rcout << "var_probs size " << prior_info[i].var_probs.size();
      //Rcout << " var_probs_short size " << prior_info[i].var_probs_short.size() << endl;
      
      // todo: var sizes adjustment
      // //DART
      // if(dart) {
      //   pi_con.dart_alpha = 1;
      //   pi_mod.dart_alpha = 1;
      //   if(var_sizes_con.size() < di_con.p) {
      //     pi_con.var_sizes.resize(di_con.p);
      //     std::fill(pi_con.var_sizes.begin(),pi_con.var_sizes.end(), 1.0/di_con.p);
      //   }
      //   if(var_sizes_mod.size() < di_mod.p) {
      //     pi_mod.var_sizes.resize(di_mod.p);
      //     std::fill(pi_mod.var_sizes.begin(),pi_mod.var_sizes.end(), 1.0/di_mod.p);
      //   }
      // }
      
      di[i] = dtemp;
      
      tree_samples ts(ntree, di[i].p, nd, di[i].basis_dim, x_info[desi]);
      final_tree_trace[i] = ts;
    }
    
    //Rcout << "Done." << endl;
    
    //--------------------------------------------------
    //setup for random effects
    size_t random_dim = random_des.n_cols;
    int nr = 1;
    if(randeff) nr = n;
    
    arma::sp_mat W = dummy_to_sparse(random_des);
    
    arma::sp_vec r(nr); //working residuals
    arma::sp_vec Wtr(random_dim); // W'r
    
    arma::sp_mat WtW = W.t()*W;//random_des.t()*random_des; //W'W
    //arma::sp_mat Sigma_inv_random = diagmat(1/(random_var_ix*random_var));
    
    arma::sp_vec tt = 1/(random_var_ix*random_var);
    arma::sp_mat Sigma_inv_random(tt.size(), tt.size());
    for(size_t i=0; i<tt.size(); ++i) Sigma_inv_random(i,i) = tt(i);
    
    // PX parameters
    arma::sp_vec eta(random_var_ix.n_cols); //random_var_ix is num random effects by num variance components
    //eta.fill(1.0);
    for(size_t i=0; i<eta.size(); ++i) {
      eta(i)=1;
    }
    
    arma::sp_mat random_var_ix_sp = dummy_to_sparse(random_var_ix);
    
    for(size_t k=0; k<nr; ++k) {
      r(k) = y[k];
      for(size_t j=0; j<num_forests; ++j) {
        r(k) -= allfits[j][k];
      }
    }
    
    Wtr = W.t()*r;
    arma::mat Wtr_tmp(Wtr);
    
    
    arma::sp_mat As = WtW/(sigma*sigma)+Sigma_inv_random;
    arma::mat A(As); // = WtW/(sigma*sigma)+Sigma_inv_random;
    arma::vec b = Wtr_tmp/(sigma*sigma);
    arma::sp_vec gamma = solve(A,b);
    arma::sp_vec allfit_random;
    if(randeff) allfit_random = W*gamma;
    
    //allfit_random.fill(0);
    
    // Setup for ordinal cutpoints ---------------------------------------
    int cutsJ = round(max(y_obs));
    arma::vec z = y_;
    arma::vec cuts = cutpoints_init;
    
    //--------------------------------------------------
    // Set fits
    double* allfit = new double[n]; //yhat
    for(size_t i=0;i<n;i++) {
      allfit[i] = 0;
      for(size_t j=0; j<num_forests; ++j) {
        allfit[i] += allfits[j][i];
      }
      if(randeff) allfit[i] += allfit_random[i];
    }
    
    // output storage
    NumericVector sigma_post(nd);
    
    // For ordinal
    NumericMatrix z_post; // latent z storage, for ordinal data
    NumericVector z_sd;
    NumericMatrix cuts_post;
    NumericMatrix ustar_post;
    if(ordinal) {
      z_post = NumericMatrix(nd,n);
      z_sd = NumericVector(nd);
      cuts_post = NumericMatrix(nd,cutsJ);
    }
    ustar_post = NumericMatrix(nd, n);
    
    //if(R_trace) forest_trace_R[s][save_ctr][j] = trees[s][j].flatten();
    std::vector<std::vector<List> > forest_trace_R(num_forests);
    for(size_t j=0; j<num_forests; ++j) {
      forest_trace_R[j].resize(nd);
      for(size_t i=0; i<nd; ++i) {
        List init_list(trees[j].size());
        forest_trace_R[j][i] = init_list;
      }
    }
    
    std::vector<NumericMatrix> forest_fits(num_forests);
    for(size_t j=0; j<num_forests; ++j) {
      NumericMatrix postfits(nd,n);
      forest_fits[j] = postfits;
    }
    
    //  NumericMatrix m_post(nd,n);
    NumericMatrix yhat_post(nd,n);
    //  NumericMatrix b_post(nd,n);
    
    NumericMatrix etas_post(nd,num_forests);
    
    
    arma::mat gamma_post(nd,gamma.n_elem);
    arma::mat random_sd_post(nd,random_var.n_elem);
    
    std::vector<NumericMatrix> split_probs(num_forests);
    for(size_t j=0; j<num_forests; ++j) {
      if(prior_info[j].dart>0) {
        split_probs[j] = NumericMatrix(nd,prior_info[j].var_probs.size());
      }
    }
    
    std::vector<arma::cube> post_coefs;
    if(return_coefs) {
      post_coefs.resize(num_forests);
      for(size_t j=0; j<num_forests; ++j) {
        arma::cube tt(di[j].basis_dim, di[j].n, nd);
        tt.fill(0);
        post_coefs[j] = tt;
      }
    }
    //
    //   arma::cube scoefs_mod(di_mod.basis_dim, di_mod.n, nd);
    //   arma::mat coefs_mod(di_mod.basis_dim, di_mod.n);
    //
    //   arma::cube scoefs_con(di_con.basis_dim, di_con.n, nd);
    //   arma::mat coefs_con(di_con.basis_dim, di_con.n);
    
    //  NumericMatrix spred2(nd,dip.n);
    
    /*
     //save stuff to tree file
     treef << xi << endl; //cutpoints
     treef << m << endl;  //number of trees
     treef << p << endl;  //dimension of x's
     treef << (int)(nd/thin) << endl;
     */
    
    //*****************************************************************************
    /* MCMC
     * note: the allfit objects are all carrying the appropriate scales
     */
    //*****************************************************************************
    
    Rcout << "\nBeginning MCMC:\n";
    time_t tp;
    int time1 = time(&tp);
    
    size_t save_ctr = 0;
    for(size_t i=0;i<(nd*thin+burn);i++) {
      
      //Rcout << allfit_con[0] << endl;
      
      if(prior_sample) {
        for(int k=0; k<n; k++) y[k] = gen.normal(allfit[k], sigma);
      }
      
      if(lower_bd.size()>1) {
        for(int k=0; k<n; k++) {
          //Rcout << y[k] << " " << allfit[k] << " " << sigma << " "<< lower_bd[k] << endl;
          if(lower_bd[k]!=-INFINITY) y[k] = rtnormlo(allfit[k], sigma, lower_bd[k]);
          // Rcout << y[k] << endl;
        }
      }
      
      if(upper_bd.size()>1) {
        for(int k=0; k<n; k++) {
          if(upper_bd[k]!=INFINITY) y[k] = -rtnormlo(-allfit[k], sigma, -upper_bd[k]);
        }
      }
      
      
      //Rcout << "latents" << endl;
      // Ordinal latent variable update
      
      if(ordinal) {
        arma::vec cuts_z = shift(arma::resize(cuts, cutsJ + 1, 1), 1); ///
        //Rcout << cuts_z;
        cuts_z(0) = - INFINITY; ///
        //Rcout << cuts_z;
        for(size_t k = 0; k < n; k++){
          double latmean = allfit[k];
          // Rcout << "loix" << y_obs[k] - 1 << endl;
          // Rcout << "hiix" << y_obs[k] << endl;
          // Rcout << cuts_z(round(y_obs[k]) - 1) << endl;
          // Rcout << cuts_z(round(y_obs[k])) << endl;
          // Rcout << "zs" << z.size() << " ys " << y.size() << endl;
          z(k) = gen.rtnormlohi(latmean, cuts_z(round(y_obs[k]) - 1), cuts_z(round(y_obs[k])));
          y[k] = z[k];
        }
      }
      
      //Rcout << "latents done" << endl;
      
      //Rcout << "a" << endl;
      Rcpp::checkUserInterrupt();
      if(i%status_interval==0) {
        Rcout << "iteration: " << i << " sigma/SD(Y): "<< sigma << endl;
      }
      
      for(size_t s=0; s<num_forests; ++s) {
        for(size_t j=0; j < trees[s].size(); ++j) {
          fit_basis(trees[s][j],x_info[s],di[s],ftemp,node_pointers[s][j],false,prior_info[s].vanilla);
          //Rcout << "fits " << s << " " << j <<endl;
          for(size_t k=0;k<n;k++) {
            if(ftemp[k] != ftemp[k]) {
              //Rcout << "tree " << j <<" obs "<< k<<" "<< endl;
              //Rcout << t_con[j] << endl;
              stop("nan in ftemp");
            }
            allfit[k] = allfit[k]-prior_info[s].eta*ftemp[k];
            allfits[s][k] = allfits[s][k]-prior_info[s].eta*ftemp[k];
            r_tree[k] = (y[k]-allfit[k])/prior_info[s].eta;
            if(r_tree[k] != r_tree[k]) {
              //Rcout << (y[k]-allfit[k]) << endl;
              //Rcout << pi_con.eta << endl;
              //Rcout << r_con[k] << endl;
              stop("NaN in resid");
            }
          }
          
          
          //Rcout << "b" << endl;
          
          if(!prior_info[s].sum_to_zero) {
            //Rcout << " bd " << endl;
            double aa = bd_basis(trees[s][j],x_info[s],di[s],prior_info[s],gen,node_pointers[s][j]);
            //Rcout << " aa " << aa << endl;
            //Rcout << " drmu" << endl;
            drmu_basis(trees[s][j],x_info[s],di[s],prior_info[s],gen);
          } else {
            
            //Rcout << " start bd "  << endl;
            double aa = bd_sumtozero(trees[s][j],x_info[s],di[s],prior_info[s],gen,node_pointers[s][j]);
            //Rcout << " aa " << aa << endl;
            
            drmu_sumtozero(trees[s][j],x_info[s],di[s],prior_info[s],gen);
            //Rcout << " drmu" << endl;
            
          }
          //Rcout << " second fit" << endl;
          //fit_basis(t_con[j],xi_con,di_con,ftemp,node_pointers_con[j],false,vanilla);
          fit_basis(trees[s][j],x_info[s],di[s],ftemp,node_pointers[s][j],false,prior_info[s].vanilla);
          //Rcout << " start allfits" << endl;
          for(size_t k=0;k<n;k++) {
            allfit[k] += prior_info[s].eta*ftemp[k];
            allfits[s][k] += prior_info[s].eta*ftemp[k];
          }
          
          //Rcout << " done allfits" << endl;
        }
      }
      
      //Rcout << "done updating trees " << endl;
      
      //
      //DART
      for(size_t s=0; s<num_forests; ++s) {
        if(prior_info[s].dart & (i>(0.25*burn))) {
          //Rcout << "DART updates" << endl;
          update_dart_noaug(trees[s],vargrps[s],varwts[s],unique_vars[s],prior_info[s], di[s], x_info[s], gen);
          //Rcout << "DART updates complete" << endl;
        }
      }
      
      //update PX parameters
      
      //Rcout << "etas" << endl;
      
      double eta_old;
      if(true) {
        for(size_t s=0; s<num_forests; ++s) {
          if(sample_eta[s]>0) {
            for(size_t k=0;k<n;k++) {
              ftemp[k] = y[k] - (allfit[k] - allfits[s][k]);
            }
            eta_old = prior_info[s].eta;
            //update_scale(ftemp, &(allfits[s])[0], n, sigma, prior_info[s], gen); <- seems to work
            update_scale(ftemp, allfits[s], n, sigma, prior_info[s], gen);
            
            //Rcout << "s = " << s << " gamma = " << prior_info[s].gamma << " eta = " << prior_info[s].eta << endl;
            
            for(size_t k=0; k<n; ++k) {
              allfit[k] -= allfits[s][k];
              allfits[s][k] = allfits[s][k] * prior_info[s].eta / eta_old;
              allfit[k] += allfits[s][k];
            }
            
            prior_info[s].sigma = sigma/fabs(prior_info[s].eta);
          }
        }
      }
      
      //Rcout << "e" << endl;
      
      if(randeff) {
        //update random effects
        for(size_t k=0; k<n; ++k) {
          r(k) = y[k] - allfit[k] + allfit_random[k];
          allfit[k] -= allfit_random[k];
        }
        
        Wtr = W.t()*r;
        
        //arma::mat adj = diagmat(random_var_ix*eta);
        
        arma::sp_mat adj = spdiag_from_vec(random_var_ix*eta);
        
        
        //    Rcout << adj << endl << endl;
        arma::sp_mat Phi = adj*WtW*adj/(sigma*sigma) + Sigma_inv_random;
        Phi = 0.5*(Phi + Phi.t());
        arma::sp_vec m = adj*Wtr/(sigma*sigma);
        //Rcout << m << Phi << endl << Sigma_inv_random;
        
        arma::vec mt(m); arma::mat Phit(Phi);
        gamma = rmvnorm_post(mt, Phit);
        
        //Rcout << "updated gamma";
        
        // Update px parameters eta
        
        //arma::sp_mat dgam = spdiag_from_vec(gamma);
        arma::vec gv(gamma);
        arma::sp_mat adj2 = spdiag_from_vec(gv)*random_var_ix_sp;
        arma::mat II = arma::eye(eta.size(), eta.size());
        arma::sp_mat Phi2 = adj2.t()*WtW*adj2/(sigma*sigma) + dummy_to_sparse(II);
        arma::sp_vec m2 = adj2.t()*Wtr/(sigma*sigma);
        Phi2 = 0.5*(Phi2 + Phi2.t());
        arma::vec m2t(m2); arma::mat Phi2t(Phi2);
        eta = rmvnorm_post(m2t, Phi2t);
        
        //Rcout << "updated eta";
        
        // Update variance parameters
        
        arma::vec ssqs   = random_var_ix.t()*(gamma % gamma);
        //Rcout << "A";
        arma::rowvec counts = sum(random_var_ix, 0);
        //Rcout << "B";
        for(size_t ii=0; ii<random_var_ix.n_cols; ++ii) {
          random_var(ii) = 1.0/gen.gamma(0.5*(random_var_df + counts(ii)), 1.0)*2.0/(random_var_df/randeff_scales(ii)*randeff_scales(ii) + ssqs(ii));
        }
        //Rcout << "updated vars" << endl;
        Sigma_inv_random = spdiag_from_vec(1/(random_var_ix*random_var));
        
        //Rcout << random_var_ix*random_var;
        
        allfit_random = W*spdiag_from_vec(random_var_ix*eta)*gamma;
        
        //is rebuilding allfits still necessary?
        for(size_t k=0; k<n; ++k) {
          allfit[k] = allfit_random(k);
          for(size_t s=0; s<num_forests; ++s) {
            allfit[k] += allfits[s][k];
          }
          //allfit[k] = allfit_con[k] + allfit_mod[k] + ; //+= allfit_random[k];
        }
      }
      
      //draw sigma
      double rss = 0.0;
      double restemp = 0.0;
      for(size_t k=0;k<n;k++) {
        restemp = y[k]-allfit[k];
        rss += restemp*restemp;
      }
      //Rcout << y[0] << " " << y[5] << endl;
      //Rcout << allfit[0] << " " << allfit[5] << endl;
      //Rcout << "rss " << rss << endl;
      if(!ordinal) sigma = sqrt((nu*lambda + rss)/gen.chi_square(nu+n));
      //pi_con.sigma = sigma/fabs(pi_con.eta);
      //pi_mod.sigma = sigma/fabs(pi_mod.eta);
      
      for(size_t s=0; s<num_forests; ++s) {
        // Rcout << "sigma " << sigma << " eta " <<prior_info[s].eta << endl;
        prior_info[s].sigma = sigma/fabs(prior_info[s].eta);
      }
      
      // sigma = 0.1;
      
      //Rcout << "cuts" << endl;
      // update cutpoints
      if(ordinal && (cutsJ > 2)) {
        arma::vec allfit_vec = arma::vec(n);
        for(int k = 0; k<n; k++) {
          allfit_vec[k] = allfit[k];
        }
        
        // current cutpoints (0, ..., infty):
        arma::vec current = arma::resize(shift(cuts,(cutsJ - 1)), cutsJ - 2 ,1);
        arma::vec g = arma::resize(shift(Newt_Raph_cuts(y_obs, allfit_vec, cuts),(cutsJ - 1)), cutsJ - 2 ,1);
        arma::vec g_full = arma::resize(g, cutsJ, 1);
        g_full(cutsJ - 2) = INFINITY;
        g_full = shift(g_full, 1);
        
        arma::mat D = arma::inv(- hessian_cuts(y_obs, allfit_vec, g_full));
        double df = 100.0; // degrees of freedom for multivariate t proposal
        arma::vec proposal = gen.rmvt_cpp(D, g, df);
        
        // compute acceptance prob.
        double a = log_post_cuts(proposal, y_obs, allfit_vec) - log_post_cuts(current, y_obs, allfit_vec);
        double b = dmvt_cpp(current, D, g, df, true) - dmvt_cpp(proposal, D, g, df, true);
        double alpha = 0;
        if(a + b <= 0){
          alpha = a + b;
        }
        alpha = exp(alpha);
        
        // accept / reject proposal:
        double p = 1;
        if(std::isnan(exp(a + b)) == false) {
          p = gen.uniform();
          if(p < alpha) {
            current = proposal;
          }
        }
        //Rcout << "alpha: " << alpha << endl;
        //Rcout << "p: " << p << endl;
        
        // lp = log_post_cuts(current, y_obs, allfit_vec); ///////////////////
        
        // resize vector to include fixed cutpoints:
        current = arma::resize(current, cutsJ, 1);
        current(cutsJ - 2) = INFINITY;
        current = shift(current, 1);
        cuts = current;
      }
      
     // Rcout << " end cuts" << endl;

    //  Rcout << Omega[0](1,1) << " " << Omega[0][1,2] << " " << Omega[0][2,2] << endl;
      
      // Update Omega 
      if(update_omega_bool){ // turned on if multibart is given a list of parameters for updating omega
        size_t ulength = 100;
        arma::vec ugrid = arma::linspace(-1, 1, ulength); // possible values of u*
        arma::vec ustar(n);
        arma::vec m(ulength);
        std::vector<int> h(n, 0); // vector of integer draws from k
        
        arma::mat Omega_grid = Omega_update(ugrid, 
                                            nb, 
                                            L, 
                                            lscale, 
                                            sig_omega);
        //Rcout << "ncol 1: " << Omega[0].n_cols << " nrow 1: " << Omega[0].n_rows << endl;
        //Rcout << "ncol: " << Omega_grid.n_cols << " nrow: " << Omega_grid.n_rows << endl;
        
        for(size_t s=0; s<1; ++s) {
          for(size_t k = 0; k < n; k++) {
            m.zeros();
            r_tree[k] = y[k] - (allfit[k] - allfits[s][k]); // update r_tree residuals for forest s
            
            m = Omega_grid * coef_basis_i(k, trees[s], x_info[s], di[s]);
            arma::vec weights = (- 0.5 * arma::square((r_tree[k] - m) / sigma)); // vector of weights for each candidate in u_grid
            h[k] = rdisc_log(weights);
            ustar(k) = ugrid(h[k]);
            
            allfit[k] -= allfits[s][k];
            allfits[s][k] = m[h[k]];
            allfit[k] += allfits[s][k];
            //Omega[s].col(k) = (Omega_grid.row(h[k])).t();
          } 
          
          arma::mat Omega_new = Omega_update(ustar, nb, L, lscale, sig_omega);
          Omega[s] = Omega_new.t();
        }
      }
      
      //Rcout << "ncol 2: " << Omega[0].n_cols << " nrow 2: " << Omega[0].n_rows << endl;
      //Rcout << "ncol Omega2: " << Omega[1].n_cols << " nrow Omega2: " << Omega[1].n_rows << endl;
      
     // Rcout << "Updated Omega" << endl;
      
      if( ((i>=burn) & (i % thin==0)) )  {
        //for(size_t j=0;j<m;j++) treef << t[j] << endl;
        
        //      msd_post(save_ctr) = fabs(pi_con.eta)*con_sd;
        //      bsd_post(save_ctr) = fabs(pi_mod.eta)*mod_sd;
        
        //pi_mod.var_probs
        
        // for(size_t j=0; j<pi_con.var_probs.size(); ++j) {
        //   var_prob_con(save_ctr, j) = pi_con.var_probs[j];
        // }
        // for(size_t j=0; j<pi_mod.var_probs.size(); ++j) {
        //   var_prob_mod(save_ctr, j) = pi_mod.var_probs[j];
        // }
        
        gamma_post.row(save_ctr) = (diagmat(random_var_ix*eta)*gamma).t();
        random_sd_post.row(save_ctr) = (sqrt( eta % eta % random_var)).t();
        
        sigma_post(save_ctr) = sigma;
        //      eta_con_post(save_ctr) = pi_con.eta;
        //      eta_mod_post(save_ctr) = pi_mod.eta;
        
        for(size_t k=0;k<n;k++) {
          //        m_post(save_ctr, k) = allfit_con[k];
          //        b_post(save_ctr, k) = allfit_mod[k];
          yhat_post(save_ctr, k) = allfit[k];
        }
        
        if(ordinal) {
          for(size_t k=0;k<n;k++) {
            z_post(save_ctr, k) = z[k];
          }
          z_sd(save_ctr) = arma::stddev(z);
          for(size_t k=0;k<cutsJ;k++) {
            cuts_post(save_ctr, k) = cuts[k];
          }
        }
        
        // Save omega latent us
        //if(TRUE) {
        //  for(size_t k=0; k<n; k++){
        //      ustar_post(save_ctr,k) = ustar(k);
        //    }
          
        //}
        
        /*
        for(size_t s=0; s<num_forests; ++s) {
          for(size_t j=0; j< trees[s].size(); ++j) {
            for(size_t ii=0; ii<n; ++ii) {
              uint32_t u = reinterpret_cast<uintptr_t>(node_pointers[s][j][ii]);
              node_pointer_trace[save_ctr][s][j][ii] = u;
            }
          }
        }
         */
        
        for(size_t s=0; s<num_forests; ++s) {
          etas_post(save_ctr,s) = prior_info[s].eta;
          //for(size_t j=0; j<num_forests; ++j) {
          if(prior_info[s].dart>0) {
            //split_probs[j] = NumericMatrix(nd,prior_info[j].var_probs.size());
            for(size_t q=0; q<prior_info[s].var_probs.size(); ++q) {
              split_probs[s](save_ctr,q) = prior_info[s].var_probs[q];
            }
          }
          //}
          for(size_t j=0; j< trees[s].size(); ++j) {
            if(return_coefs) post_coefs[s].slice(save_ctr) += prior_info[s].eta*coef_basis(trees[s][j], x_info[s], di[s]);
            //if(text_trace) tree_streams[s] << trees[s][j];
            final_tree_trace[s].t[save_ctr][j] = trees[s][j];
            final_tree_trace[s].t[save_ctr][j].compress();
            final_tree_trace[s].t[save_ctr][j].scale(prior_info[s].eta);
            //if(R_trace) forest_trace_R[s][save_ctr][j] = trees[s][j].flatten(prior_info[s].eta);
          }
        }
        
        save_ctr += 1;
      }
    }
    
    int time2 = time(&tp);
    Rcout << "time for loop: " << time2 - time1 << endl;
    
    delete[] allfit;
    delete[] ftemp;
    
    std::vector<Rcpp::RawVector> Rtree_serial_streams(num_forests);
    
    
    //   std::stringstream ss;
    //   {
    //     cereal::BinaryOutputArchive oarchive(ss); // Create an output archive
    //     oarchive(my_instance);
    //   }
    //   ss.seekg(0, ss.end);
    //   RawVector retval(ss.tellg());
    //   ss.seekg(0, ss.beg);
    //   ss.read(reinterpret_cast<char*>(&retval[0]), retval.size());
    //   return retval;
    // }
    
    if(text_trace) {
      for(size_t s=0; s<num_forests; ++s) {
        Rtree_streams[s] = final_tree_trace[s].save_string();//tree_streams[s].str();
      }
    }
    
    if(R_trace) {
      for(size_t s=0; s<num_forests; ++s) {
        {
          cereal::BinaryOutputArchive oarchive(serial_streams[s]); // Create an output archive
          oarchive(final_tree_trace[s]); // Write the data to the archive
        }
        serial_streams[s].seekg(0, serial_streams[s].end);
        RawVector retval(serial_streams[s].tellg());
        serial_streams[s].seekg(0, serial_streams[s].beg);
        serial_streams[s].read(reinterpret_cast<char*>(&retval[0]), retval.size());
        Rtree_serial_streams[s] = retval;
      }
    }
    //
    // {
    //   cereal::BinaryOutputArchive oarchive(treef_serial); // Create an output archive
    //   oarchive(final_tree_trace[0]); // Write the data to the archive
    // }
    
    //if(text_trace) treef << tree_streams[0].rdbuf();
    //treef.close();
    //treef_serial.close();
    
    return(List::create(_["yhat_post"] = yhat_post,
                        _["coefs"] = post_coefs,
                        _["etas"] = etas_post,
                        _["sigma"] = sigma_post, //_["msd"] = msd_post, _["bsd"] = bsd_post,
                        _["gamma"] = gamma_post,
                        _["random_sd_post"] = random_sd_post,
                        _["tree_streams"] = Rtree_streams,
                        //_["tree_serials"] = Rtree_serial_streams,
                        _["tree_trace"] = final_tree_trace,
                        _["split_probs"] = split_probs,
                        //_["y_last"] = y,
                        _["z_trace"] = z_post,
                        _["z_sd"] = z_sd,
                        _["cuts_trace"] = cuts_post//,
                        //_["ustar_trace"] = ustar_post
                        //_["npr"] = node_pointer_trace
    ));
}

#ifndef GUARD_bd_h
#define GUARD_bd_h

#include "rng.h"
#include "info.h"
#include "tree.h"
#include "funs.h"

double prop_death(tree& x, xinfo& xi, pinfo& pi, tree::npv& goodbots, double& PBx, 
                  tree::tree_p& nx, size_t& v, size_t& c, 
                  RNG& gen);
double prop_birth(tree& x, xinfo& xi, pinfo& pi, tree::npv& goodbots, double& PBx, 
                  tree::tree_p& nx, size_t& v, size_t& c, 
                  RNG& gen);

double bd_basis(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen, std::vector<tree::tree_cp>& node_pointers);
double bd_sumtozero(tree& x, xinfo& xi, dinfo& di, pinfo& pi, RNG& gen, std::vector<tree::tree_cp>& node_pointers);

#endif

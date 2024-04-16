make_bart_designs = function(X_list, basis_matrix_list, groups_list = NA) {
  num_designs = length(X_list)
  if(num_designs != length(basis_matrix_list)) {
    stop("X_list and basis_matrix_list must be the same size")
  }
  
  if(all(!is.na(groups_list))) {
    group = sapply(groups_list, function(x) length(x)>1)
    out = lapply(1:num_designs, function(i) make_bart_design(X = X_list[[i]], basis_matrix = basis_matrix_list[[i]], 
                                                             group = group[i], groups = groups_list[[i]], index = i-1))
  } else {
    out = lapply(1:num_designs, function(i) make_bart_design(X = X_list[[i]], basis_matrix = basis_matrix_list[[i]], index = i-1))
  }
  
  return(out)
}

make_bart_design = function(
  X,
  basis_matrix,
  model_mat = matrix(NA),
  vargroups = attr(X, "grp"),
  varwts    = attr(X, "wt"),
  group = FALSE,
  groups = 0,
  index = -1
  ) {
  
  cutpoint_list = lapply(1:ncol(X), function(i) multibart:::.cp_quantile(X[,i]))
  
  if(is.null(vargroups)) vargroups = 1:ncol(X)
  if(is.null(varwts))    vargroups = rep(1,ncol(X))
  
  Qt = matrix(0)
  R = matrix(0)
  if(nrow(model_mat)>1) {
    decomp = qr(model_mat)
    Qt = t(qr.Q(decomp))
    R = qr.R(decomp)
  }
  
  list(X=t(X),
       Omega = t(basis_matrix),
       Qt = Qt,
       R = R,
       info = cutpoint_list,
       index = index,
       group = group,
       groups = groups,
       vargroups = vargroups-1,
       varwts    = varwts,
       unique_vars = length(unique(vargroups))
  )
}

make_bart_spec = function(
  design,
  ntree,
  Sigma0,
  mu0 = NA,
  scale_df = -1,
  vanilla = FALSE,
  alpha = 0.95, 
  beta = 2,
  dart = FALSE,
  update_leaf_scale = TRUE,
  sum_to_zero = FALSE,
  nosplits = FALSE,
  ortho = FALSE
  ) {
  
  if(is.na(mu0)) mu0 = rep(0, ncol(Sigma0))
  if(ncol(design$Omega)==1 & all( (design$Omega - 1)<1e-8)) vanilla = TRUE
  
  list(
    design_index = design$index,
    ntree = ntree,
    scale_df = scale_df,
    mu0 = mu0,
    Sigma0 = Sigma0,
    vanilla = vanilla,
    alpha = alpha,
    beta = beta,
    dart = dart,
    sample_eta = ifelse(update_leaf_scale, 1, -1),
    sum_to_zero = ifelse(sum_to_zero, 1, -1),
    nosplits = ifelse(nosplits, 1, -1),
    ortho  = ifelse(ortho, 1, -1)
  )
}
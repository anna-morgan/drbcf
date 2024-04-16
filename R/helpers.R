.ident <- function(...){
  # courtesy https://stackoverflow.com/questions/19966515/how-do-i-test-if-three-variables-are-equal-r
  args <- c(...)
  if( length( args ) > 2L ){
    #  recursively call ident()
    out <- c( identical( args[1] , args[2] ) , .ident(args[-1]))
  }else{
    out <- identical( args[1] , args[2] )
  }
  return( all( out ) )
}

.cp_quantile = function(x, num=10000, cat_levels=8){
  nobs = length(x)
  nuniq = length(unique(x))
  
  if(nuniq==1) {
    ret = x[1]
    warning("A supplied covariate contains a single distinct value.")
  } else if(nuniq < cat_levels) {
    xx = sort(unique(x))
    ret = xx[-length(xx)] + diff(xx)/2
  } else {
    q = approxfun(sort(x),quantile(x,p = 0:(nobs-1)/nobs))
    ind = seq(min(x),max(x),length.out=num)
    ret = q(ind)
  }
  
  return(ret)
}


#' Construct BART model matrix from a data frame or factor/numeric vector.
#' 
#' Adapted from bartModelMatrix in the BART package. This version only constructs
#' the design matrix, adds grouping information as attributes, and converts
#' character columns to factors.
#'
#' @param X 
#'
#' @return A matrix with attributes indicating which columns are dummies constructed from a single factor
#' @export
#'
#' @examples
mb_modelmatrix = function(X) {
  
  X.class = class(X)[1]
  
  if(X.class=='factor') {
    X.class='data.frame'
    X=data.frame(X=X)
  }
  
  grp = NULL
  varwt = NULL
  
  if(X.class=='data.frame') {
    p=dim(X)[2]
    xnm = names(X)
    for(i in 1:p) {
      #if(is.character(X[[i]])) X[[i]] = factor(X[[i]])
      if(is.factor(X[[i]]) | is.character(X[[i]])) {
        Xtemp = class.ind(as.factor(X[[i]]))
        if(ncol(Xtemp)==2) Xtemp = Xtemp[,-2,drop=FALSE]
        colnames(Xtemp) = paste(xnm[i],1:ncol(Xtemp),sep='.')
        X[[i]]=Xtemp
        grp=c(grp, rep(i, ncol(Xtemp)))
        varwt=c(varwt, rep(1/ncol(Xtemp), ncol(Xtemp)))
      } else {
        X[[i]]=cbind(X[[i]])
        colnames(X[[i]])=xnm[i]
        grp = c(grp, i)
        varwt = c(varwt,1)
      }
    }
    Xtemp=cbind(X[[1]])
    if(p>1) for(i in 2:p) Xtemp=cbind(Xtemp, X[[i]])
    X=Xtemp
  }
  else if(X.class=='numeric' | X.class=='integer') {
    X=cbind(as.numeric(X))
    grp=1
  }
  else if(X.class=='NULL') return(X)
  else if(X.class!='matrix')
    stop('Expecting either a factor, a vector, a matrix or a data.frame')
  
  
  if(X.class=='matrix') {
    grp = 1:ncol(X)
    varwt = rep(1, ncol(X))
  }
    
  attr(X, "grp") = grp
  attr(X, "wt")  = varwt
  return(X)
}

# Add default metadata for DART updates if it's missing
mb_dart_default_meta = function(X) {
  if(is.null(attr(X, "grp"))) attr(X, "grp") = rep(1, ncol(X))
  if(is.null(attr(X, "wt")))  attr(X, "wt") = rep(1, ncol(X))
  return(X)
}

# cbind X and y while extending attributes for DART
mb_dart_cbind = function(X, y) {
  ext_n = 1
  if(is.matrix(y)) ext_n = ncol(y)
  X = mb_dart_default_meta(X)
  y = mb_dart_default_meta(as.matrix(y))
  xgrp = attr(X, "grp")
  xwt  = attr(X, "wt")
  ygrp = max(xgrp) + attr(y, "grp")
  ywt  = attr(y, "wt")
  
  ret = cbind(X,y)
  attr(ret, "grp") = c(xgrp, ygrp)
  attr(ret, "wt") = c(xwt, ywt)
  
  return(ret)
}

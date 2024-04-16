#' multibart
#'
#' Description of your package
#'
#' @docType package
#' @import Rcpp
#' @importFrom stats approxfun lm qchisq quantile sd
#' @importFrom Rcpp evalCpp
#' @useDynLib multibart
#' @name multibart
#' @export tree_samples 
NULL

loadModule("tree_samples", TRUE)
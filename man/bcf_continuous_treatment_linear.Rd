% Generated by roxygen2: do not edit by hand
% Please edit documentation in R/bcf_continuous_linear.R
\name{bcf_continuous_treatment_linear}
\alias{bcf_continuous_treatment_linear}
\title{Fit Bayesian Causal Forests}
\usage{
bcf_continuous_treatment_linear(
  y,
  z,
  x_control,
  x_moderate = x_control,
  zhat = rep(0.5, length(y)),
  randeff_design = matrix(1),
  randeff_variance_component_design = matrix(1),
  randeff_scales = 1,
  randeff_df = 3,
  nburn,
  nsim,
  nthin = 1,
  update_interval = 100,
  ntree_control = 250,
  sd_control = 2 * sd(y),
  base_control = 0.95,
  power_control = 2,
  ntree_moderate = 50,
  sd_moderate = 0.25 * sd(y)/sd(z),
  base_moderate = 0.25,
  power_moderate = 3,
  nu = 3,
  lambda = NULL,
  sigq = 0.9,
  sighat = NULL,
  include_zhat = "control",
  use_muscale = TRUE,
  use_tauscale = TRUE,
  debug = FALSE
)
}
\arguments{
\item{y}{Response variable}

\item{z}{Treatment assignments}

\item{x_control}{Design matrix for the "prognostic" function mu(x)}

\item{x_moderate}{Design matrix for the covariate-dependent treatment effects tau(x)}

\item{zhat}{Length n estimates of E(Z|X)}

\item{randeff_design}{XX}

\item{randeff_variance_component_design}{XX}

\item{randeff_scales}{XX}

\item{randeff_df}{XX}

\item{nburn}{Number of burn-in MCMC iterations}

\item{nsim}{Number of MCMC iterations to save after burn-in}

\item{nthin}{Save every nthin'th MCMC iterate. The total number of MCMC iterations will be nsim*nthin + nburn.}

\item{update_interval}{Print status every update_interval MCMC iterations}

\item{ntree_control}{Number of trees in mu(x)}

\item{sd_control}{SD(mu(x)) marginally at any covariate value (or its prior median if use_muscale=TRUE)}

\item{base_control}{Base for tree prior on mu(x) trees (see details)}

\item{power_control}{Power for the tree prior on mu(x) trees}

\item{ntree_moderate}{Number of trees in tau(x)}

\item{sd_moderate}{SD(tau(x)) marginally at any covariate value (or its prior median if use_tauscale=TRUE)}

\item{base_moderate}{Base for tree prior on tau(x) trees (see details)}

\item{power_moderate}{Power for the tree prior on tau(x) trees (see details)}

\item{nu}{Degrees of freedom in the chisq prior on \eqn{sigma^2}}

\item{lambda}{Scale parameter in the chisq prior on \eqn{sigma^2}}

\item{sigq}{Calibration quantile for the chisq prior on \eqn{sigma^2}}

\item{sighat}{Calibration estimate for the chisq prior on \eqn{sigma^2}}

\item{include_zhat}{Takes values "control", "moderate", "both" or "none". Whether to
include zhat in mu(x) ("control"), tau(x) ("moderate"), both or none. Values of "control"
or "both" are HIGHLY recommended with observational data.}

\item{use_muscale}{Use a half-Cauchy hyperprior on the scale of mu.}

\item{use_tauscale}{Use a half-Normal prior on the scale of tau.}

\item{debug}{XX}
}
\value{
A list with elements
\item{tau}{\code{nsim} by \code{n} matrix of posterior samples of individual treatment effects}
\item{mu}{\code{nsim} by \code{n} matrix of posterior samples of individual treatment effects}
\item{sigma}{Length \code{nsim} vector of posterior samples of sigma}
}
\description{
Fit Bayesian Causal Forests
}
\details{
Fits a generalized version of the Bayesian Causal Forest model (Hahn et. al. 2018): For a response
variable y, treatment z, and covariates x,
\deqn{y_i = \mu(x_i, \hat z_i) + \tau(x_i, \pi_i)\omega(z_i) + \epsilon_i}
where \eqn{\z_i} is an (optional) estimate of \eqn{E(Z_i | X_i=x_i)} and
\eqn{\epsilon_i \sim N(0,\sigma^2)}

Some notes:
\itemize{
   \item x_control and x_moderate must be numeric matrices. See e.g. the makeModelMatrix function in the
   dbarts package for appropriately constructing a design matrix from a data.frame
   \item sd_control and sd_moderate are the prior SD(mu(x)) and SD(tau(x)) at a given value of x (respectively). If
   use_muscale = FALSE, then this is the parameter \eqn{\sigma_\mu} from the original BART paper, where the leaf parameters
   have prior distribution \eqn{N(0, \sigma_\mu/m)}, where m is the number of trees.
   If use_muscale=TRUE then sd_control is the prior median of a half Cauchy prior for SD(mu(x)). If use_tauscale = TRUE,
   then sd_moderate is the prior median of a half Normal prior for SD(tau(x)).
   \item By default the prior on \eqn{\sigma^2} is calibrated as in Chipman, George and McCulloch (2008).


}
}
\examples{
\donttest{

#TODO: Update this example
# data generating process
p = 3 #two control variables and one moderator
n = 250
#
set.seed(1)

x = matrix(rnorm(n*p), nrow=n)

# create targeted selection
q = -1*(x[,1]>(x[,2])) + 1*(x[,1]<(x[,2]))

# generate treatment variable
pi = pnorm(q)
z = rbinom(n,1,pi)

# tau is the true (homogeneous) treatment effect
tau = (0.5*(x[,3] > -3/4) + 0.25*(x[,3] > 0) + 0.25*(x[,3]>3/4))

# generate the response using q, tau and z
mu = (q + tau*z)

# set the noise level relative to the expected mean function of Y
sigma = diff(range(q + tau*pi))/8

# draw the response variable with additive error
y = mu + sigma*rnorm(n)

# If you didn't know pi, you would estimate it here
zhat = pnorm(q)

bcf_fit = bcf(y, z, x, x, zhat, nburn=2000, nsim=2000)

# Get posterior of treatment effects
tau_post = bcf_fit$tau
tauhat = colMeans(tau_post)
plot(tau, tauhat); abline(0,1)

}
\dontshow{

# data generating process
p = 3 #two control variables and one moderator
n = 250
#
set.seed(1)

x = matrix(rnorm(n*p), nrow=n)

# create targeted selection
q = -1*(x[,1]>(x[,2])) + 1*(x[,1]<(x[,2]))

# generate treatment variable
pi = pnorm(q)
z = rbinom(n,1,pi)

# tau is the true (homogeneous) treatment effect
tau = (0.5*(x[,3] > -3/4) + 0.25*(x[,3] > 0) + 0.25*(x[,3]>3/4))

# generate the response using q, tau and z
mu = (q + tau*z)

# set the noise level relative to the expected mean function of Y
sigma = diff(range(q + tau*pi))/8

# draw the response variable with additive error
y = mu + sigma*rnorm(n)

}
}
\references{
Hahn, Murray, and Carvalho(2017). Bayesian regression tree models for causal inference: regularization, confounding, and heterogeneous effects.
 https://arxiv.org/abs/1706.09523. (Call citation("bcf") from the
command line for citation information in Bibtex format.)
}

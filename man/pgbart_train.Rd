\name{pgbart_train}
\alias{pgbart_train}
\title{Train of particle Gibbs for Bayesian Additive Regression Trees Sampler}
\description{
    Creates a sampler object for a given problem which fits a particle Gibbs Bayesian Additive Regression Trees model.
}
\usage{
pgbart_train(train_data, train_label,
             if_test = FALSE,
             test_data = matrix(nrow = 0, ncol = 0),
             test_label = vector(length = 0),
             model_file,
             alpha_bart = 3.0, q_bart = 0.9,
             alpha_split = 0.95, beta_split = 0.5,
             if_center_label = FALSE,
             ess_threshold = 1.0,
             init_seed_id = 1, if_set_seed = TRUE,
             k_bart = 2.0, m_bart = 1,
             min_size = 1,ndpost = 1000, nskip = 100,keepevery = 1,
             verbose_level = 0, n_particles = 10,
             )

}
\arguments{
\item{train_data}{Explanatory variables for training (in sample) data.\cr
May be a matrix,with (as usual) rows corresponding to observations and columns to variables.\cr}

\item{train_label}{Dependent variable for training (in sample) data.\cr
   \code{train_label} is a always a vector .\cr}

\item{if_test}{Logical;if TRUE have a test.}

\item{test_data}{Explanatory variables for test (out of sample) data.\cr
   Should have same structure as \code{train_data}.\cr
   \code{bart} will generate draws of \eqn{f(x)} for each \eqn{x} which is a row of \code{train_data}.}


\item{test_label}{Dependent variable for testing (in sample) data.\cr
  \code{test_label} is a always a vector .\cr
}

\item{model_file}{The path to save model file. The file contains a lot of details about the trees and nodes, even information about every node. }

\item{alpha_bart}{Degrees of freedom for error variance prior. \cr
The hyperparameters \code{alpha_bart} and \code{q_bart} indirectly control the shape and rate of the inverse gamma prior over \eqn{\sigma^2}.\cr}

\item{q_bart}{
 The quantile of the prior that the rough estimate is placed at.
   The closer the quantile is to 1,
   the more aggresive the fit will be as you are putting more prior weight
   on error standard deviations (\eqn{\sigma}) less than the rough estimate.
   Not used if y is binary.

The hyperparameters \code{alpha_bart} and \code{q_bart} indirectly control the shape and rate of the inverse gamma prior over \eqn{\sigma^2}\cr  needs to be in [0, 1], default 0.9}

\item{alpha_split}{alpha_split for cgm tree prior.A hyperparameter between 0 and 1,default 0.95\cr Higher values of \code{alpha_bart} lead to deeper trees while higher
values of \code{1/beta_split} lead to shallower trees.\cr}

\item{beta_split}{A hyperparameter between 0 and 1.\cr beta_split for cgm tree prior  Higher values of \code{alpha_bart} lead to deeper trees while higher
values of \code{1/beta_split} lead to shallower trees.\cr
}

\item{if_center_label}{The idea is that \eqn{f} is shrunk towards 0, so the offset allows you to shrink towards
   a probability other than 0.5.}

\item{ess_threshold}{The threshold of ESS - effective sample size.\cr
ess_threshold needs to be in [0, 1], default 1.0\cr
ESS is a measure of how well the chain mixes and is frequently used to assess performance of MCMC algorithms. }

\item{init_seed_id}{Set the initial value of random seed.}

\item{if_set_seed}{Logical controlling whether or not setting the initial value of random seed.If FALSE,this value of seed will be in the default value.}

\item{k_bart}{For numeric train_label,\code{k_bart} is the number of prior standard deviations \eqn{E(Y |x) = f(x)} is away from +/-0.5.
  k_bart controls the prior over mu (mu_prec) in BART, it must be positive, default 2.0}

\item{m_bart}{The number of regression trees\cr it must be not less than 1, default 1}

\item{min_size}{min_size is minimum number of data points at leaf nodes, it must be not less than 1, default 1}

\item{ndpost}{
     The number of posterior draws after burn in, \code{ndpost / keepevery} will actually be returned.
   }

   \item{nskip}{
     Number of MCMC iterations to be treated as burn in.
   }

 \item{keepevery}{
     Every \code{keepevery} draw is kept to be returned to the user. Useful for \dQuote{thinning}
     samples.
   }


\item{verbose_level}{level of detail about the intermediate process,verbosity level (0 is minimum, 1 is maximum), default 1 }

\item{n_particles}{The number of particle,it must be not less than 1, default 10\cr}


}
\details{
   BART is an Bayesian MCMC method.
   At each MCMC interation, we produce a draw from the joint posterior
   \eqn{(f,\sigma) | (x,y)}{(f,sigma) \| (x,y)} in the numeric \eqn{y} case
   and just \eqn{f} in the binary \eqn{y} case.

   Thus, unlike a lot of other modelling methods in R, we do not produce a single model object
   from which fits and summaries may be extracted.  The output consists of values
   \eqn{f^*(x)}{f*(x)} (and \eqn{\sigma^*}{sigma*} in the numeric case) where * denotes a particular draw.
   The \eqn{x} is either a row from the training data (train_data) or the test data (test_data).
}
\value{
\code{bart} returns a list assigned class \code{bart}. For applicable
  quantities, \code{ndpost / keepevery} samples are returned.
  In the numeric \eqn{y} case, the list has components:

  \item{\code{yhat.train}}{
        A matrix of posterior samples. The \eqn{(i, j)} value is \eqn{f^*(x_j)}{f*(x_j)}
        for the \eqn{i}th kept draw of of the posterior of \eqn{f}
        and the \eqn{j}th row of \code{x.train}.}
  \item{\code{yhat.test}}{
      Same as \code{yhat.train} but now the \eqn{x}s are the rows of the test data.}
  \item{\code{yhat.train.mean}}{
        Mean of \code{yhat.train} columns.}
  \item{\code{yhat.test.mean}}{
        Mean of \code{yhat.test} columns.}
  \item{\code{sigma}}{
        Posterior samples of \code{sigma}, the residual/error standard deviation.}
  \item{\code{first.sigma}}{
        Burn-in draws of \code{sigma}.}
  \item{\code{varcount}}{
        A matrix with number of rows equal to the number of kept draws and each column
        corresponding to a training variable. Contains the total count of the number of
        times that variable is used in a tree decision rule (over all trees).}

  Output file depends on the type of verbose.It may contain a lot of train detail,such as node information , every result of every iteration and so on.
  However,\code{mse_train} and  \code{loglik_train} will be returned all the time.
}
\references{
Balaji Lakshminarayanan, Daniel M. Roy, E., Yee Whye Teh(2015)
   Particle Gibbs for Bayesian Additive Regression Trees.
   \emph{Artificial Intelligence and Statistics }.

Chipman, H., George, E., and McCulloch R. (2010)
   Bayesian Additive Regression Trees.
   \emph{The Annals of Applied Statistics}, \bold{4,1}, 266-298.
}

\author{
xxxx: \email{xx.xx@xx.com}\cr
yyy: \email{yy.yy@yy.com}.
}

\examples{
##simulate data (example from Friedman MARS paper)
f = function(x){
10*sin(pi*x[,1]*x[,2]) + 20*(x[,3]-.5)^2+10*x[,4]+5*x[,5]
}
sigma = 1.0  #y = f(x) + sigma*z , z~N(0,1)
n = 100      #number of observations
set.seed(99)
x=matrix(runif(n*10),n,10) #10 variables, only first 5 matter
Ey = f(x)
y=Ey+sigma*rnorm(n)
pgbart_train(train_data = x, train_label = y, model_file = "./pgbart.model", if_test = FALSE) #the output path is locale by default.
}

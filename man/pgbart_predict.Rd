\name{pgbart_predict}
\alias{pgbart_predict}
\title{Test of particle Gibbs for Bayesian Additive Regression Trees Sampler}
\description{
    Creates a sampler object for a given problem which fits a particle Gibbs for Bayesian Additive Regression Trees model.
}
\usage{
pgbart_predict(test_data, model_file)
}
\arguments{
\item{test_data}{Explanatory variables for test (out of sample) data.\cr
   Should have same structure as x.train.\cr
   \code{bart} will generate draws of \eqn{f(x)} for each \eqn{x} which is a row of \code{train_data}.}
\item{model_file}{The path to save model file. The file contains a lot of details about the trees and nodes, even information about every node. }
}
\value{
\item{\code{yhat}}{
        A maxtix of predict result. the \eqn{i} th row is a predict result of \code{x.test}. And the \eqn{j} th column is the whole predict result for the \eqn{j} th point of \code{x.test} }
  \item{\code{yhat.mean}}{
        Mean of \code{yhat} columns.}
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

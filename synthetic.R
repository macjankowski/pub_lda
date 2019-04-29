library(Compositional)
library(MCMCpack)

generateWord <- function(theta, beta){
  K <- length(theta)
  V <- dim(beta)[2]
  z <- sample(1:K, 1, replace=TRUE, prob=theta)
  sample(1:V, 1, replace=TRUE, beta[z,])
}

generateDocument <- function(V, K, Nd, alpha, beta){
  
  theta <- rdirichlet(1, rep(alpha,K))
  document <- rep(0, V)
  
  for(i in 1:Nd){
    word <- generateWord(theta, beta)
    document[word] <- document[word] + 1
  }

  document
}

# M - number of documents
# K - number of topics
# alpha - dirichlet parameter for generating theta
# delta - dirichlet parameter for generating beta
# lambda - parameter of poisson for generating lengths of documents
generateCorpus <- function(V, M, K, alpha, delta, lambda){
  N <- rpois(M, lambda)
  
  if(is.scalar(delta)){
    delta <- rep(delta, V)
  }
  
  beta <- rdirichlet(K, alpha=delta)
  
  corpus <- sapply(1:M, function(i){
    generateDocument(V, K, N[i], alpha, beta)
  })
  t(corpus)
}

estimateDirichletParamsFromData <- function(data){
  diri.est(data)
}

is.scalar <- function(x) is.atomic(x) && length(x) == 1L


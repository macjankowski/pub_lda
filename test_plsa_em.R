library(testthat)
source('./plsa_em.R')


check_theta <- function(theta){
  
  M <- length(theta[,1])
  expected <- rep(1, M)
  expect_equal(apply(theta, MARGIN = c(1), sum), expected)
}

check_beta <- function(beta){
  
  K <- length(beta[,1])
  
  for(k in 1:K){
    expect_equal(sum(beta[k,]), 1)
  }

}

check_gamma <- function(gamma){
  
  start.time <- Sys.time()
  V <- dim(gamma)[1]
  M <- dim(gamma)[2]
  K <- dim(gamma)[3]
  
  expect_equal(apply(gamma, MARGIN = c(1,2), sum),  matrix(1, V, M))
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  #print("Execution of check_gamma")
  #print(time.taken)
}

test_reestimate_gamma <- function(K, tf) {
  
  theta <-initTheta(K, tf)
  check_theta(theta)

  beta <-initBeta(K, tf)
  check_beta(beta)
  
  gamma <- reestimate_gamma(theta, beta)
  check_gamma(gamma)
  gamma
}


#theta[m,k] = tf[i,m] * gamma[k,i,m]
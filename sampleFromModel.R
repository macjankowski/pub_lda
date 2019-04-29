library(Compositional)
library(MCMCpack)

generateDocumentFromModel <- function(V, Nd, beta, theta){
  
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
generateCorpusFromModel <- function(model, docLengths){
  
  thetas <- model$ldaTrainData
  M <- model$topicmodel@Dim[1]
  V <- model$topicmodel@Dim[2]
  K <- model$topicmodel@k
  beta <- exp(model$topicmodel@beta)

  generateCorpusFromParameters(K=K, V=V, M=M, thetas=thetas, beta=beta, docLengths=docLengths)
}

generateCorpusFromParameters <- function(K, V, M, thetas, beta, docLengths){
  
  corpus <- sapply(1:M, function(i){
    theta <- thetas[i,]
    generateDocumentFromModel(V, docLengths[i], beta, theta)
  })
  
  print(paste("Generated synthetic corpus for ",K," topics"))
  t(corpus)
}

generateCorpusFromModelSubset <- function(model, docLengths, subsetForClassIndex){
  
  thetas <- model$ldaTrainData[subsetForClassIndex,]
  M <- dim(thetas)[1]
  V <- model$topicmodel@Dim[2]
  K <- model$topicmodel@k
  beta <- exp(model$topicmodel@beta)
  
  corpus <- sapply(1:M, function(i){
    theta <- thetas[i,]
    generateDocumentFromModel(V, docLengths[i], beta, theta)
  })
  
  print(paste("Generated synthetic corpus for ",K," topics"))
  t(corpus)
}


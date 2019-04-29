


createPerClassLda <- function(K, trainLabels, trainTf){
  
  labels <- unique(trainLabels)
  lapply(labels, function(l){
    subsetForClass <- which(trainLabels==l)
    ldaModel <- LDA(
      x = trainTf[subsetForClass,], 
      k=K, 
      control=list(seed=SEED)
    )
    trainSetForClass <- posterior(ldaModel)[2]$topics
    list(topicmodel=ldaModel, ldaTrainData=trainSetForClass)
  })
}

classifyUsingPerplexity <- function(models, testMatrix){
  
  nRows <- dim(testMatrix)[1]
  response <- rep(0, nRows)
  for(i in 1:nRows){
    doc <- testMatrix[i,]
    p_1 <- perplexity(models[[1]], doc)
    p_2 <- perplexity(models[[2]], doc)
    response[i] <- which.min(c(p_1,p_2))-1
  }
  response
}

classifyUsingLikelihood <- function(models, testMatrix){
  
  nRows <- dim(testMatrix)[1]
  response <- rep(0, nRows)

  thetas_0 <- models[[1]]$ldaTrainData
  betas_0 <- exp(models[[1]]$topicmodel@beta)
  log_multDist_0 <- log(estimateMultinomialFromCorpus(thetas_0 %*% betas_0))
  
  thetas_1 <- models[[2]]$ldaTrainData
  betas_1 <- exp(models[[2]]$topicmodel@beta)
  log_multDist_1 <- log(estimateMultinomialFromCorpus(thetas_1 %*% betas_1))
  
  for(i in 1:nRows){
    doc <- testMatrix[i,]

    l1 <- sum(log_multDist_0 * as.vector(doc))
    l2 <- sum(log_multDist_1 * as.vector(doc))

    response[i] <- which.min(c(l1,l2))-1
  }
  response
}

calculateErrorRate <- function(testLabels, predictedLabels){
  count <- length(testLabels)
  
  respXor <- xor(predictedLabels, as.integer(testLabels))
  length(respXor[respXor == TRUE])/count
}




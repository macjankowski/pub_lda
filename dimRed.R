library("topicmodels")
library("lsa")
library("svs")

SEED <- 2100

calculateLDA <- function(tfData, topic_number){
  
  start.time <- Sys.time()
  
  topicmodel <- LDA(tfData$cleanedTrainMatrix, k=topic_number, control=list(seed=SEED))
  trainData <- posterior(topicmodel)[2]$topics
  testData <- posterior(topicmodel, tfData$cleanedTestMatrix)[2]$topics
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(topicmodel=topicmodel, ldaTrainData=trainData, ldaTestData=testData, duration=duration)
}

calculateLSA <- function(tfidfData, topic_number){
  
  start.time <- Sys.time()
  
  lsa_train_tdidf <- createLSATrain(tfidfData$cleanedTrainMatrix, dims=topic_number)
  lsa.training.set.tfidf = lsa_train_tdidf$matrix
  
  lsa_test_tdidf <- createLSATest(tfidfData$cleanedTestMatrix, lsa_train_tdidf$lsa)
  lsa.testing.set.tfidf = lsa_test_tdidf$matrix
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(lsaTrainData=lsa.training.set.tfidf, lsaTestData=lsa.testing.set.tfidf, duration=duration)
}

createLSATrain <- function(rawTrainMatrix, dims=dimcalc_share(share=0.8)){
  start.time <- Sys.time()
  m=as.matrix(rawTrainMatrix)
  dim(m)
  lsa_m=lsa(t(m),dims)
  dim(lsa_m$tk)
  dim(lsa_m$dk)
  length(lsa_m$sk)
  
  lsa.training.set = t(diag(lsa_m$sk) %*% t(lsa_m$dk))
  dim(lsa.training.set)
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(matrix=lsa.training.set, lsa=lsa_m, duration=duration)
}

createLSATest <- function(rawTestMatrix, lsa_m){
  
  start.time <- Sys.time()
  m_test=as.matrix(rawTestMatrix)
  dim(t(m_test))
  
  lsa.testing.set <- t(diag(1,dim(lsa_m$tk)[2]) %*% t(lsa_m$tk) %*% t(m_test))
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(matrix=lsa.testing.set, duration=duration)
}

createPLSA <- function(tfData, topic_number){
  fast_plsa(tfData, topic_number)
}


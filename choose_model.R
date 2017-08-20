
library("ldatuning")

source('./dimRed.R')
source('./classification.R')




printf <- function(...) cat(sprintf(...), sep='\n', file=stderr())

estimateTopicsCount4Methods <- function(from,to,step, tfidfData, methods = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014")){
  
  estimateTopicsCount4MethodsRange(range = seq(from = from, to = to, by = step), 
                              tfidfData = tfidfData, methods = methods)
  
}

estimateTopicsCount4MethodsRange <- function(range, tfidfData, methods = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014")){

  
  start.time <- Sys.time()
  
  result <- FindTopicsNumber(
    tfidfData$cleanedTrainMatrix,
    topics = range,
    metrics = methods,
    method = "Gibbs",
    control = list(seed = 77),
    mc.cores = 4L,
    verbose = TRUE
  )
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(ldatuningResults=result, duration=duration)
}

estimateTopicsCountLSA <- function(from,to,step, tfidfData, tree_number){
  
  start.time <- Sys.time()
  topicCounts <- seq(from,to,step)
  
  runLsa <- function(k) {
    runLsaForKTopics(k, tfidfData, tree_number)
  }
  
  results <- lapply(topicCounts, runLsa)
  errors <- lapply(results, function(x){x$error})

  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(rfResult=errors, duration=duration)
}

runLsaForKTopics <- function(k, tfidfData, tree_number){
  
  start.time <- Sys.time()
  
  printf("Calculating lda fot %d topics\n",k)
  lsa <- calculateLSA(tfidfData, k)
  
  printf("Training lsa took  %f seconds\n",lsa$duration)

  error <- trainAndPredictSimple(tree_number=tree_number, 
                                 trainData=lsa$lsaTrainData, trainLabels=tfidfData$cleanedTrainLabels, 
                                 testData=lsa$lsaTestData, testLabels=tfidfData$cleanedTestLabels)
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(error=error, topicCount=k, duration=lsa$duration)
}

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
  
  printf("Calculating lsa fot %d topics\n",k)
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

geometricMean <- function(x){
  exp(mean(log(x)))
}

avg_mi_single_model <- function(data, labels){
  f <- function(i){ 
    d <- as.numeric(discretize(data[,i], method = "frequency", categories = 25, onlycuts=FALSE))
    mi.plugin(rbind(d, as.numeric(labels))) #+ mi.plugin(rbind(as.numeric(labels), d)) 
  }
  
  mis <- sapply(1:length(data[1,]), f)
  
  #mis <- as.numeric(discretize(data, method = "frequency", categories = 25, onlycuts=FALSE))
  
  print(mean(mis))
  mean(mis)
}

avg_mi_all_models <- function(ldaModels, labels){
  sapply(1:length(ldaModels), function(i){avg_mi_single_model(ldaModels[[i]]$ldaTrainData, labels)})
}

 

avg_spearman_single_model_without_mean <- function(data, labels){
  f <- function(i){ 
    cor.test(data[,i], as.numeric(labels), method="spearman")$estimate
  }
  
  sapply(1:length(data[1,]), f)
}

avg_spearman_single_model <- function(data, labels){
  f <- function(i){ 
    cor.test(data[,i], as.numeric(labels), method="spearman")$estimate
  }
  
  max(abs((harmonic.mean(sapply(1:length(data[1,]), f)))))
}

avg_spearman_all_models <- function(ldaModels, labels){
  sapply(1:models_count, function(i){avg_spearman_single_model(ldaModels[[i]]$ldaTrainData, labels)})
}

avgEntropyForModel <- function(ldaModel){
  posterior <- posterior(ldaModel$topicmodel)
  #entropy.plugin(posterior$terms)
  N <- length(posterior$terms)
  mean(apply(posterior$terms, 2, function(doc){
    entropy.plugin(doc)
  }))
}

maxEntropyForModel <- function(ldaModel){
  posterior <- posterior(ldaModel$topicmodel)
  N <- length(posterior$terms)
  max(apply(posterior$terms, 2, function(doc){
    entropy.plugin(doc)
  }))
}

minEntropyForModel <- function(ldaModel){
  posterior <- posterior(ldaModel$topicmodel)
  N <- length(posterior$terms)
  min(apply(posterior$terms, 2, function(doc){
    entropy.plugin(doc)
  }))
}

avgEntropyForPosterior1 <- function(ldaModel){
  posterior <- posterior(ldaModel$topicmodel)
  dim(posterior$topics)
  N <- length(posterior$topics)
  mean(apply(posterior$topics, 1, function(doc){entropy.plugin(doc)}))
  #entropy.plugin(posterior$topics)
}

avgEntropyForPosterior2 <- function(ldaModel){
  posterior <- posterior(ldaModel$topicmodel)
  dim(posterior$topics)
  #mean(apply(posterior$topics, 1, entropy.plugin))
  entropy.plugin(posterior$topics)
}



predictWithScoreUsingRfModel <- function(rfModel, testData){
  out <- predict(rfModel, as.matrix(testData), type="prob")
  best.score <- as.double(apply(out,1, max))
  best.class <- apply(out,1, which.max)-1
  b <- as.numeric(best.class)
  list(class=b, score=best.score)
}

predictWithScoreEnsemble <- function(rfModels, ldaModels){
  
  c <- list()
  s <- list()
  
  for(i in (1:length(rfModels))){
    prediction <- predictWithScoreUsingRfModel(rfModels[[i]], ldaModels[[i]]$ldaTestData)
    c[[i]] <- prediction$class
    s[[i]] <- rescale(prediction$score)
  }
  
  classes <- do.call(rbind, c)
  scores <- do.call(rbind, s)
  list(classes=classes, scores=scores)
}

predictWithScoreEnsembleSubstractMean <- function(testLabels, rfModels, ldaModels){
  
  c <- list()
  s <- list()
  
  for(i in (1:length(rfModels))){
    prediction <- predictWithScoreUsingRfModel(rfModels[[i]], ldaModels[[i]]$ldaTestData)
    c[[i]] <- prediction$class
    s[[i]] <- prediction$score - mean(prediction$score)
  }
  
  classes <- do.call(rbind, c)
  scores <- do.call(rbind, s)
  list(classes=classes, scores=scores)
}

bestClassWithScoreInEnsemble <- function(classes, scores){
  
  length <- length(classes[1,])
  
  best_classes <- sapply(1:length, function(i){
    best_class <- classes[,i]
    best_score <- scores[,i]
    
    df <- data.frame(v1=best_class, v2=best_score)
    agg <- aggregate(x=df$v2, by=list(df$v1), FUN=sum)
    agg
    bc <- as.numeric(agg[which.max(agg$x),][1])
    bc
  })
  best_classes
}

errorForEnsembleWithScoreResult <- function(classes, scores, testLabels){
  best.class.ensemble <- bestClassWithScoreInEnsemble(classes,scores)
  test.result.ensemble <- best.class.ensemble[best.class.ensemble == testLabels]
  error <- 1-length(test.result.ensemble)/length(testLabels)
  error
}

# histogram of winning documents for models

recognizedClassesForHistogram <- function(testLabels, rfModels, ldaModels, topicNumbers){
  
  p <- predictWithScoreEnsemble(rfModels, ldaModels)
  
  modelsHist <- sapply(1:length(testLabels), function(i){
    idx <- which(p$classes[,i] == testLabels[i])
    best_score <- p$scores[,i]
    best_score[-idx] = -1
    which.max(best_score)
  })
  
  modelsTopicsHist <- sapply(modelsHist, function(i){topicNumbers[i]})
  
  hist(modelsTopicsHist, breaks = 0:max(topicNumbers), xlab = "Number of topics", ylab = "Frequency", main="" )
  #qplot(modelsTopicsHist, geom="histogram", binwidth = 2.5)
}




predictUsingRfModel <- function(rfModel, testData){
  out <- predict(rfModel, as.matrix(testData), type="prob")
  best.class <- apply(out,1, which.max)-1
  b <- as.numeric(best.class)
  b
}

predictEnsemble <- function(rfModels, ldaModels){
  
  l <- list()
  
  for(i in (1:length(rfModels))){
    b <- predictUsingRfModel(rfModels[[i]], ldaModels[[i]]$ldaTestData)
    l[[i]] <- b
  }
  
  res <- do.call(rbind, l)
  res
}

mostCommon <- function(v){
  as.numeric(names(table(v)[which.max(table(v))])) 
}

bestClassInEnsemble <- function(m){
  apply(m,2, mostCommon)
}

errorForEnsembleResult <- function(res, testLabels){
  best.class.ensemble <- bestClassInEnsemble(res)
  test.result.ensemble <- best.class.ensemble[best.class.ensemble == testLabels]
  error <- 1-length(test.result.ensemble)/length(testLabels)
  error
}


# tfidfData$cleanedTestMatrix
analyseEnsemble <- function(rfModels, ldaModes, labels, testMatrix){
  
  resPredictEnsemble <- predictEnsemble(appsRfModels[subRange], appsLdaModels[subRange])
  mvError <- errorForEnsembleResult(resPredictEnsemble, testLabels)
  
  resPredictEnsembleWithScore <- predictWithScoreEnsemble(rfModels, ldaModes)
  wmvError <- errorForEnsembleWithScoreResult(resPredictEnsembleWithScore$classes, resPredictEnsembleWithScore$scores, labels)
  
  chosenModels <- chooseModelUsingPerplexity(testMatrix, ldaModes)
  errorPerplexityEnsemble(chosenModels, appsLdaModels, appsRfModels)
  
  list(mvError, wmvError, perpError)
}





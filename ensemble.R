
predictUsingRfModel <- function(rfModel, testData){
  out <- predict(rfModel, as.matrix(testData), type="prob")
  best.class <- apply(out,1, which.max)-1
  b <- as.numeric(best.class)
  b
}

predictEnsemble <- function(testLabels, rfModels, ldaModels){
  
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



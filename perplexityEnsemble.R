

chooseModelUsingPerplexity <- function(testDocumentTermMatrix, ldaModels){
  
  nRows <- testDocumentTermMatrix$nrow
  chosenModels <- rep(0,nRows)
  for(i in 1:nRows){
    document <- testDocumentTermMatrix[i,]
    perplexities_for_doc <- lapply(ldaModels, function(m){
      p <- perplexity(m$topicmodel, document)
      p
    })
    idx <- which.min(perplexities_for_doc)
    print(paste('idx = ',idx))
    chosenModels[i] <- idx
  }
  chosenModels
}

errorPerplexityEnsemble <- function(chosenModels, ldaModels, rfModels, testLabels){
  allRows <- 0
  errors <- 0
  print(paste("Number of models",length(ldaModels)))
  for(i in 1:length(ldaModels)){
    #print(paste("Loop for ",i))
    idxForIthModel <- which(chosenModels == i)
    #print(idxForIthModel)
    print(ldaModels[[i]]$topicmodel@k)
    ldaTestData <- ldaModels[[i]]$ldaTestData[idxForIthModel,] #get data for ith model
    
    nRowsChosen <- dim(ldaTestData)[1]
    print(nRowsChosen)
    if(is.null(nRowsChosen)){
      ldaTestData = matrix(ldaTestData, nrow = 1)
      nRowsChosen = 1
    }
    if(nRowsChosen > 0){
      error <- predictSimple(rfModels[[i]], ldaTestData, testLabels[idxForIthModel])
      errors = errors + (error * nRowsChosen)
      allRows = allRows + nRowsChosen
    }
  }
  #print(allRows)
  errors / allRows
}


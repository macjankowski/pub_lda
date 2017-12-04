

perModelPerplexity <- 

  
chooseModelUsingPerplexity <- function(testDocumentTermMatrix){
  
  nRows <- testDocumentTermMatrix$nrow
  chosenModels <- rep(0,380)
  for(i in 1:nRows){
    document <- tfidfData$cleanedTestMatrix[i,]
    perplexities_for_doc <- lapply(lda_models_200, function(m){
      p <- perplexity(m$topicmodel, document)
      p
    })
    chosenModels[i] <- which.min(perplexities_for_doc)
  }
  chosenModels
}




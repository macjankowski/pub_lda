library(EMT)


estimateMultinomialFromCorpus <- function(corpusMatrix){
  if(is.null(dim(corpusMatrix))){
    if(length(corpusMatrix) > 0){
      mult <- corpusMatrix / sum(corpusMatrix)
    }else{
      stop('corpusMatrix is empty')
    }
  }else{
    mult <- colSums(corpusMatrix)/sum(corpusMatrix)
  }
  mult
}

observedCountsFromCorpus <- function(corpusMatrix){
  colSums(corpusMatrix)
}

correctProb <- function(multDist){
  probOfNotAppearingToken <- 0.0000000001
  zeroProbIndices <- which(multDist == 0)
  multDistCorrected <- multDist
  multDist[zeroProbIndices] <-  probOfNotAppearingToken
  multDist
}


calculateKlBetweenDataAndSample <- function(originalTrainTf, models){
  
  originalTrainMatrix <- as.matrix(originalTrainTf)
  dataMultinomialDistribution <- estimateMultinomialFromCorpus(originalTrainMatrix)

  docLengths <- rowSums(originalTrainMatrix)
  
  samples <- lapply(models, function(m){
    generateCorpusFromModel(m, docLengths)
  })
  
  multinomialDistributionsOfSamples <- lapply(samples, estimateMultinomialFromCorpus)
  
  KlDataVsSample <- unlist(lapply(multinomialDistributionsOfSamples, function(sampleMultinomial){
    KL.plugin(dataMultinomialDistribution, sampleMultinomial)
  }))
  KlDataVsSample
}

calculatePValueBetweenDataAndSample <- function(originalTrainTf, models, samples){
  
  originalTrainMatrix <- as.matrix(originalTrainTf)
  dataMultinomialDistribution <- estimateMultinomialFromCorpus(originalTrainMatrix)
  
  dataObservedCountsOfSamples <- lapply(samples, colSums)
  
  pValues <- unlist(lapply(dataObservedCountsOfSamples, function(observedCountOfSample){
    chisq.test(observedCountOfSample, p = dataMultinomialDistribution)$p.value
  }))
  pValues
}

calculatePValueBetweenDataAndSampleForClass <- function(originalTrainTf, models, samples, labels){
  
  subsetForClassIndex <- which(labels==0)
  originalTrainMatrix <- as.matrix(originalTrainTf)
  dataMultinomialDistribution <- estimateMultinomialFromCorpus(originalTrainMatrix[subsetForClassIndex, ])
  
  dataObservedCountsOfSamples <- lapply(samples, function(sample){
    colSums(sample[subsetForClassIndex, ])
  })
  
  chi <- lapply(dataObservedCountsOfSamples, function(observedCountOfSample){
    chisq.test(observedCountOfSample, p = dataMultinomialDistribution)
  })
  chi
}

calculatePerClassPValueBetweenDataAndSample <- function(originalTrainTf, models, labels, classLabel){

  subsetForClassIndex <- which(labels==classLabel)
  trainMatrix <- as.matrix(originalTrainTf)
  trainMatrixForClass <- trainMatrix[subsetForClassIndex, ]
  multDistForClass <- estimateMultinomialFromCorpus(trainMatrixForClass)
  
  docLengths <- rowSums(trainMatrixForClass)
  
  samples <- lapply(models, function(m){
    generateCorpusFromModelSubset(m, docLengths, subsetForClassIndex)
  })
  
  dataObservedCountsOfSamples <- lapply(samples, colSums)
  
  pValues <- unlist(lapply(dataObservedCountsOfSamples, function(observedCountOfSample){
    chisq.test(observedCountOfSample, p = multDistForClass)$p.value
  }))
  pValues
}

calculateKlPerClassForModel <- function(classIndex, labels, originalTrainTf, sampleFromModel){
  subsetForClassIndex <- which(labels==classIndex)
  trainMatrix <- as.matrix(originalTrainTf)
  trainMatrixForClass <- trainMatrix[subsetForClassIndex, ]
  
  trainDim <- dim(trainMatrixForClass)
  klDiv <- if(!is.null(trainDim) && (trainDim[1] == 0)){
    0
  }else{
    multDistForClass <- estimateMultinomialFromCorpus(trainMatrixForClass)
    multDistForClassCorrected <- correctProb(multDistForClass)
    
    sampleFromModelForClass <- sampleFromModel[subsetForClassIndex, ]
    
    sampleFromModelForClassMultDist <- estimateMultinomialFromCorpus(sampleFromModelForClass)
    
    sampleFromModelForClassMultDistCorrected <- correctProb(sampleFromModelForClassMultDist)
    
    #mi.plugin(rbind(multDistForClassCorrected, sampleFromModelForClassMultDistCorrected))
    KL.plugin(multDistForClassCorrected, sampleFromModelForClassMultDistCorrected) 
  }
  klDiv
}

calculatePValuePerClassForModel <- function(classIndex, labels, originalTrainTf, sampleFromModel){
  subsetForClassIndex <- which(labels==classIndex)
  trainMatrix <- as.matrix(originalTrainTf)
  trainMatrixForClass <- trainMatrix[subsetForClassIndex, ]
  
  trainDim <- dim(trainMatrixForClass)
  pValue <- if(!is.null(trainDim) && (trainDim[1] == 0)){
    0
  }else{
    multDistForClass <- estimateMultinomialFromCorpus(trainMatrixForClass)

    sampleFromModelForClass <- sampleFromModel[subsetForClassIndex, ]
    
    observedCountsForSampled <- colSums(sampleFromModelForClass)
    
    t = chisq.test(observedCountsForSampled, p = multDistForClass)$p.value
  }
  pValue
}



calculateAvgKl <- function(range, samplesFromModel, labels, tfMatrix){
  avgKl <- rep(0, length(range))
  for(i in 1:length(range)){
    sampleData <- samplesFromModel[[i]]
    
    classes <- unique(labels)
    sumKlForSampleFromModel <- 0
    for(j in classes){
      print(paste("model = ",range[i],", class = ",j))
      kl_aaa <- calculateKlPerClassForModel(classIndex = j, labels = labels, originalTrainTf = tfMatrix,
                                            sampleFromModel = sampleData)
      sumKlForSampleFromModel <- sumKlForSampleFromModel +  kl_aaa
    }
    avgKlPerModel <- sumKlForSampleFromModel/length(classes)
    avgKl[i] = avgKlPerModel
    print(avgKlPerModel)
  }
  avgKl
}

calculateLikelihoodPerClassForModel <- function(classIndex, labels, originalTrainTf, sampleFromModel){
  subsetForClassIndex <- which(labels==classIndex)
  trainMatrix <- as.matrix(originalTrainTf)
  trainMatrixForClass <- trainMatrix[subsetForClassIndex, ]
  
  trainDim <- dim(trainMatrixForClass)

    multDistForClass <- estimateMultinomialFromCorpus(trainMatrixForClass)
    multDistForClassCorrected <- correctProb(multDistForClass)
    log_multDistForClassCorrected <- log(multDistForClassCorrected)
    
    sampleFromModelForClass <- sampleFromModel[subsetForClassIndex, ]
    
    likelihoods <- apply(sampleFromModelForClass, 1, function(row){
      log_multDistForClassCorrected * row # calculate likelihood using p^{observed counts}
    })
    
    # sum them up
    perClassLikelihood <- sum(likelihoods)
    perClassLikelihood
}

modelPerClassLikelihood <- function(labels, originalTrainTf, sampleFromModel){
  
  lik_0 <- calculateLikelihoodPerClassForModel(0, labels, originalTrainTf, sampleFromModel)
  lik_1 <- calculateLikelihoodPerClassForModel(1, labels, originalTrainTf, sampleFromModel)
  
  lik_0 + lik_1
}

calculateLikelihoodForModel <- function(labels, originalTrainTf, sampleFromModel){
  trainMatrix <- as.matrix(originalTrainTf)

  trainDim <- dim(trainMatrix)
  
  multDist <- estimateMultinomialFromCorpus(trainMatrix)
  multDistCorrected <- correctProb(multDist)
  log_multDistCorrected <- log(multDistCorrected)
  
  likelihoods <- apply(sampleFromModel, 1, function(row){
    sum(log_multDistCorrected * row) # calculate likelihood using p^{observed counts}
  })
  
  # sum them up
  logLikelihood <- sum(likelihoods)
  logLikelihood
}


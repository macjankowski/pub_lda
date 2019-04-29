

toLdaFormat <- function(trainTf){
  m <- as.matrix(trainTf)
  nRows <- dim(m)[1]
  nCols <- dim(m)[2]
  
  lst <- vector(mode = "list", length = nRows)
  
  for(i in 1:nRows){
    row <- m[i,]
    l <- length(row[row>0])
    idx <- which(row>0)
    newRow <- array(as.integer(0),c(2,l))
    for(j in 1:length(idx)){
      newRow[1,j] <- as.integer(idx[j]-1)
      newRow[2,j] <- as.integer(row[idx[j]])
    }
    lst[[i]] <- newRow
  }
  lst
}

loadClassSLDAPosterior <- function(path, truelabelsPath, withProbs=TRUE){
  
  gamma <- loadGamma(file = paste(path,"/inf-gamma.dat", sep = ""))
  theta <- gammaToTheta(gamma)
  
  if(withProbs){
    predictedProbsPath = paste(path,"/inf-probs.dat", sep = "")
    probs <- loadProbabilities(file = predictedProbsPath)
  }
  
  
  predictedLabelsPath = paste(path,"/inf-labels.dat", sep = "")
  predictedLabels <- loadLabels(file=predictedLabelsPath)
  
  trueLabels <- loadLabels(file=truelabelsPath)
  
  errorRate <- calculateErrorRate(trueLabels, predictedLabels)
  
  if(withProbs){
    list(theta=theta, labels=predictedLabels, errorRate=errorRate, probabilities=probs)
  }else{
    list(theta=theta, labels=predictedLabels, errorRate=errorRate)
  }
}



readModels <- function(rootPath, topicCounts, truelabelsPath){
  models <- lapply(topicCounts, function(topicCount){
    modelPath <- paste(rootPath,'/',topicCount, sep = "")
    loadClassSLDATrainData(modelPath, truelabelsPath)
  })
  models
}

readPrediction <- function(rootPath, topicCounts, truelabelsPath, withProbs=TRUE){
  models <- lapply(topicCounts, function(topicCount){
    inferencePath <- paste(rootPath,'/',topicCount, sep = "")
    loadClassSLDAPosterior(inferencePath, truelabelsPath, withProbs)
  })
  models
}

loadModelWeight <- function(path){
  data <- read.table(path, sep = "" , header = FALSE)
  as.numeric(data)
}

loadModelsWeights <- function(rootDir, modelsIds){
  
  sapply(modelsIds, function(id){
    path <- paste(rootDir,'/',id,'/model_weight_in_ensemble.txt', sep = "")
    loadModelWeight(path)
  })
  
}

loadClassSLDATrainData <- function(path, labelsPath){

  gamma <- loadGamma(file = paste(path,"/final.gamma", sep = ""))
  theta <- gammaToTheta(gamma)
  labels <- loadLabels(file=labelsPath)
  list(theta=theta, labels=labels)
}

loadGamma <- function(file){
  data <- read.table(file, sep = " " , header = FALSE)
  data
}

loadProbabilities <- function(file){
  data <- read.table(file, sep = " " , header = FALSE)
  data
}

loadLabels <- function(file){
  data <- read.table(file, header = FALSE)
  as.matrix(data)[,1]
}


gammaToTheta <- function(gamma){
  theta <- t(apply(gamma, 1 ,function(row){
    row/sum(row)
  }))
  theta
}

calculateErrorRate <- function(trueLabels, predictedLabels){
  incorrectCount <- length(trueLabels[trueLabels != predictedLabels])
  allCount <- length(trueLabels)
  errorRate <- incorrectCount/allCount
  errorRate
}

calculateErrorRateForEnsemble <- function(boostWithProbs, model_weights){
  L <- length(boostWithProbs)
  M <- length(boostWithProbs[[1]]$probabilities[,1])
  C <- length(boostWithProbs[[1]]$probabilities[1,])
  
  qube <- array(0, dim=c(M, L, C))
  
  for(l in 1:L){
    qube[,l,] <- as.matrix(boostWithProbs[[l]]$probabilities) * model_weights[l]
  }
  
  classProbsInEnsemble <- apply(qube,c(1,3),sum)
  
  predictedLabelsForEnsemble <- apply(classProbsInEnsemble,1,which.max)-1
  calculateErrorRate(trueLabels = smsTrueLabels, predictedLabels = predictedLabelsForEnsemble)
}

calculateErrorRateForEnsembleAdaBoost <- function(boostWithProbs, model_weights, trueLabels){
  L <- length(boostWithProbs)
  calculateErrorRateForEnsembleAdaBoostSubset(boostWithProbs, model_weights, trueLabels, L)
}

calculateErrorRateForEnsembleAdaBoostSubset <- function(boostWithProbs, model_weights, trueLabels, L){
  M <- length(boostWithProbs[[1]]$probabilities[,1])
  C <- length(boostWithProbs[[1]]$probabilities[1,])
  
  qube <- array(0, dim=c(M, L, C))
  
  for(l in 1:L){
    two_values <- boostWithProbs[[l]]$probabilities
    two_values[two_values > 0.5] <- 1
    two_values[two_values <= 0.5] <- -1
    
    qube[,l,] <- as.matrix(two_values) * model_weights[l]
  }
  
  classProbsInEnsemble <- apply(qube,c(1,3),sum)
  
  predictedLabelsForEnsemble <- apply(classProbsInEnsemble,1,which.max)-1
  calculateErrorRate(trueLabels = trueLabels, predictedLabels = predictedLabelsForEnsemble)
}

analyseMutualInformationSLDA <- function(range, models, labels, header="", bins=10, 
                                     rescale=TRUE, runOnTest = FALSE, discretizeMethod="frequency"){
  
  mutualInfoCompact <- function(model) {
    avg_mi_single_model_2_on_data(model$theta, labels, bins=bins)
  }
  
  mutualInfoCompactForPosterior <- unlist(lapply(models, mutualInfoCompact))
  errorsUnscaled <- sapply(models, function(m){m$errorRate})

  if(rescale){
    mutualInfoCompactForPosterior <- rescale(mutualInfoCompactForPosterior)
    errors <- rescale(errorsUnscaled)
  }
  
  miWithClassificationError <- data.frame(range, mutualInfoCompactForPosterior, errors)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError, id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Mutual Information", "sLDA error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

analyseConditionalEntropySLDA <- function(range, models, labels, header="", bins=2, 
                                      rescale=TRUE, runOnTest = FALSE, discretizeMethod="frequency"){
  
  conditionalEntropyCompact <- function(model) {
    conditionalEntropyOnData(model$theta, labels, bins=2)
  }
  
  condEntropyForPosterior <- unlist(lapply(models, conditionalEntropyCompact))
  errorsUnscaled <- sapply(models, function(m){m$errorRate})
  
  
  if(rescale){
    condEntropyForPosterior <- rescale(condEntropyForPosterior)
    errors <- rescale(errorsUnscaled)
  }
  
  miWithClassificationError <- data.frame(range, condEntropyForPosterior, errors)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError,id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Conditional entropy", "sLDA error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

analyseBoosting <- function(range, rootInferencePath, rootModelWeightsPath, trueLabelsPath, header=""){
  
  inference <- readPrediction(rootPath=rootInferencePath,
                              topicCounts = range-1,
                              truelabelsPath = trueLabelsPath)
  
  trueLabels <- loadLabels(file=trueLabelsPath)
  
  m_weights <- loadModelsWeights(rootModelWeightsPath,range-1)
  
  cumErrors <- unlist(lapply(range, function(i){
    err <- calculateErrorRateForEnsembleAdaBoostSubset(inference, m_weights, trueLabels, i)
    err
  }))
  
  infer_errors <- sapply(inference, function(m){m$errorRate})
  
  boostErrorDf <- data.frame(range, infer_errors, cumErrors)
  boostErrorDfMelted <- melt(data = boostErrorDf, id.vars = "range")
  
  dev.new()
  ggplot(data = boostErrorDfMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Single model error", "Ensemble error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Boosting iterations", y="Test Error", color="Variable") + theme_bw()  + #theme(text = element_text(size=30)) + #theme_classic(base_size = 18)+
    ggtitle(header) + ylim(0, 0.7)

}

analyseModelForDifferentK <- function(topicNumbers, rootInferencePath, rootInferenceTrainPath, trueTestLabelsPath, trueTrainLabelsPath, header="", ensembleError) {
  inference <- readPrediction(
    rootPath=rootInferencePath,
    topicCounts=topicNumbers,
    truelabelsPath=trueTestLabelsPath,
    withProbs=FALSE
  )
  
  inferenceOnTrain <- readPrediction(
    rootPath=rootInferenceTrainPath,
    topicCounts=topicNumbers,
    truelabelsPath=trueTrainLabelsPath, 
    withProbs=FALSE
  )
  
  errsPredTrain <- sapply(inferenceOnTrain, function(p){ p$errorRate })
  
  errsPred <- sapply(inference, function(p){ p$errorRate })

  ens_error <- sapply(errsPred, function(x){ensembleError})
  
  aaaDf <- data.frame(topicNumbers, errsPred, errsPredTrain, ens_error)
  aaaDfMelted <- melt(data = aaaDf, id.vars = "topicNumbers")
  
  dev.new()
  ggplot(data = aaaDfMelted, aes(x = topicNumbers, y = value, 
    color = factor(variable, labels = c("Single model error on test", "Single model error on train", "Ensemble error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topic number", y="Test Error", color="Variable") + theme_bw()  + #theme_classic(base_size = 18)+
    ggtitle("") + ylim(0, 0.5)
}

library(MCMCpack)
library(scales)
library(reshape2)
library(ggplot2)


analysePosteriorEntropy <- function(range, models, header=""){
  
  avgPosteriorEntropy <- unlist(lapply(models, avgEntropyForPosterior1))
  expectedEntropiesList <- sapply(models, function(m){
    alpha <- m$topicmodel@alpha
    K <- m$topicmodel@k
    expectedEntropy(alpha, K)
  })
  expectedEntropies <- unlist(expectedEntropiesList)
  maxEntropy <- sapply(range, log)
  maxEntropyWithPosterior <- data.frame(range,  
                                        avgPosteriorEntropy, 
                                        maxEntropy,
                                        expectedEntropies)
  maxEntropyWithPosteriorMelted <- melt(data = maxEntropyWithPosterior, id.vars = "range")
  dev.new()
  legend <- function(v) {
    factor(v, labels = c("posterior entropy", "max entropy", "expected entropy"))
  }
  ggplot(data = maxEntropyWithPosteriorMelted, aes(x = range, y = value, 
    color = legend(variable), linetype=legend(variable))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable", linetype="Variable") + theme_classic(base_size = 18) +
    ggtitle(header)
}

analyseJoinedEntropy <- function(range, models, labels, rfErrors, sVMErrors, header="", bins=10){
  
  joinedEntropyCompact <- function(model) {
    joinedEntropy(model, labels, bins=bins)
  }

  joinedEntropyForPosterior <- unlist(lapply(models, joinedEntropyCompact))
  joinedEntropyForPosteriorRescaled <- rescale(joinedEntropyForPosterior)

  rfErrorsVectorRescaled <- rescale(rfErrors)
  sVMErrorsVectorRescaled <- rescale(sVMErrors)
  
  print(paste('range: ',length(range)))
  print(paste('joinedEntropyForPosteriorRescaled: ',length(joinedEntropyForPosteriorRescaled)))
  print(paste('rfErrorsVectorRescaled: ',length(rfErrorsVectorRescaled)))
  print(paste('sVMErrorsVectorRescaled: ',length(sVMErrorsVectorRescaled)))
  

  miWithClassificationError <- data.frame(range,  joinedEntropyForPosteriorRescaled,
                                                    rfErrorsVectorRescaled,
                                                    sVMErrorsVectorRescaled)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError, 
                                                               id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Joined entropy", "Random Forest error", "SVM Error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

analyseConditionalEntropy <- function(range, models, labels, rfErrors, sVMErrors, header="", bins=10, 
                                      rescale=FALSE, runOnTest = FALSE, discretizeMethod="frequency"){
  
  conditionalEntropyCompact <- function(model) {
    conditionalEntropy(model, labels, bins=bins, runOnTest = runOnTest, discretizeMethod=discretizeMethod)
    #conditionalEntropy2(model, labels, bins=bins)
  }

  entropyOfResponse <- infotheo::entropy(labels,method="emp")
  condEntropyForPosterior <- unlist(lapply(models, conditionalEntropyCompact))

  if(rescale){
    condEntropyForPosterior <- rescale(condEntropyForPosterior)
    rfErrors <- rescale(rfErrors)
    sVMErrors <- rescale(sVMErrors)
  }
  
  miWithClassificationError <- data.frame(range, condEntropyForPosterior,
                                          rfErrors,
                                          sVMErrors)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError, 
                                           id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Conditional entropy", "Random Forest error", "SVM Error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

analyseConditionalEntropyNoSvm <- function(range, models, labels, rfErrors, header="", bins=10, rescale = FALSE, runOnTest = FALSE, discretizeMethod="frequency"){
  
  conditionalEntropyCompact <- function(model) {
    conditionalEntropy(model, labels, bins=bins, runOnTest = runOnTest, discretizeMethod=discretizeMethod)
  }
  
  condEntropyForPosterior <- unlist(lapply(models, conditionalEntropyCompact))
  
  if(rescale){
    condEntropyForPosterior <- rescale(condEntropyForPosterior)
    rfErrors <- rescale(rfErrors)
  }

  condEntropyWithClassificationError <- data.frame(range, condEntropyForPosterior,
                                                   rfErrors)
  
  condEntropyWithClassificationErrorMelted <- melt(data = condEntropyWithClassificationError, 
                                           id.vars = "range")
  
  dev.new()
  ggplot(data = condEntropyWithClassificationErrorMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Conditional entropy", "Random Forest error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

analyseJoinedEntropyNoSvm <- function(range, models, labels, rfErrors, header="", bins=10){
  
  joinedEntropyCompact <- function(model) {
    joinedEntropy(model, labels, bins=bins)
  }
  
  conditionalEntropyCompact <- function(model) {
    conditionalEntropy(model, labels, bins=bins)
  }
  
  joinedEntropyForPosterior <- unlist(lapply(models, joinedEntropyCompact))
  joinedEntropyForPosteriorRescaled <- rescale(joinedEntropyForPosterior)
  
  condEntropyForPosterior <- unlist(lapply(models, conditionalEntropyCompact))
  condEntropyForPosteriorRescaled <- rescale(condEntropyForPosterior)
  
  rfErrorsVectorRescaled <- rescale(rfErrors)

  print(paste('range: ',length(range)))
  print(paste('joinedEntropyForPosteriorRescaled: ',length(joinedEntropyForPosteriorRescaled)))
  print(paste('rfErrorsVectorRescaled: ',length(rfErrorsVectorRescaled)))
  
  miWithClassificationError <- data.frame(range,  joinedEntropyForPosteriorRescaled,
                                          rfErrorsVectorRescaled)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError, 
                                           id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
     color = factor(variable, labels = c("Joined entropy", "Random Forest error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

expectedEntropy <- function(alpha, dim){
  cat <- rdirichlet(100, rep(alpha, dim))
  mean(apply(cat, 1, function(doc){entropy.plugin(doc)}))
}

analyseAverageEntropy <- function(range, models, rfErrors, svmErrors, header="", rescale=TRUE){
  
  avgEntropyForModels <- unlist(lapply(models, avgEntropyForModel))
  avgEntropyForModels
  
  if(rescale){
    rfErrors <- rescale(rfErrors)
    svmErrors <- rescale(svmErrors)
    avgEntropyForModels <- rescale(avgEntropyForModels)
  }
  
  avgEntropyOfTopicsWithError <- data.frame(range,  avgEntropyForModels,
                                            rfErrors, svmErrors)
  
  avgEntropyOfTopicsWithErrorMelted <- melt(data = avgEntropyOfTopicsWithError, 
                                            id.vars = "range")
  
  dev.new()
  ggplot(data = avgEntropyOfTopicsWithErrorMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Average entropy of topics", "Random Forest error", "SVM error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
}

analyseAverageEntropyNoSvm <- function(range, models, rfErrors, header=""){

  avgEntropyForModels <- lapply(models, avgEntropyForModel)
  avgMultinomialEntropyForModels <- lapply(models, avgMultinomialEntropyForModel)
  avgEntropyForModels <- rescale(unlist(avgEntropyForModels))
  avgEntropyForModels

  rfErrors <- rescale(rfErrors)
  
  avgEntropyOfTopicsWithError <- data.frame(range,  avgEntropyForModels,
                                            rfErrors)
  
  avgEntropyOfTopicsWithErrorMelted <- melt(data = avgEntropyOfTopicsWithError, 
                                           id.vars = "range")
  
  dev.new()
  ggplot(data = avgEntropyOfTopicsWithErrorMelted, aes(x = range, y = value, 
    color = factor(variable, labels = c("Average entropy of topics", "Random Forest error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
}

analyseMutualInformation <- function(range, models, labels, rfErrors, sVMErrors, header="", bins=10, 
                                      rescale=TRUE, runOnTest = FALSE, discretizeMethod="frequency"){
  
  mutualInfoCompact <- function(model) {
    #mutualInfo(model, labels, bins=bins, discretizeMethod=discretizeMethod)
    avg_mi_single_model_2(model, labels, bins=bins)
  }
  
  entropyOfResponse <- infotheo::entropy(labels,method="emp")
  mutualInfoCompactForPosterior <- unlist(lapply(models, mutualInfoCompact))
  #mutualInfoCompactForPosterior <-  avg_mi_all_models(models, labels)
  
  if(rescale){
    mutualInfoCompactForPosterior <- rescale(mutualInfoCompactForPosterior)
    rfErrors <- rescale(rfErrors)
    sVMErrors <- rescale(sVMErrors)
  }
  
  miWithClassificationError <- data.frame(range, mutualInfoCompactForPosterior,
                                          rfErrors,
                                          sVMErrors)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError, 
                                           id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
                                                      color = factor(variable, labels = c("Mutual Information", "Random Forest error", "SVM Error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

analyseMutualInformationNoSvm <- function(range, models, labels, rfErrors, header="", bins=10, 
                                     rescale=TRUE, runOnTest = FALSE, discretizeMethod="frequency"){
  
  mutualInfoCompact <- function(model) {
    mutualInfo(model, labels, bins=bins, discretizeMethod=discretizeMethod)
  }
  
  entropyOfResponse <- infotheo::entropy(labels,method="emp")
  mutualInfoCompactForPosterior <- unlist(lapply(models, mutualInfoCompact))
  #mutualInfoCompactForPosterior <-  avg_mi_all_models(models, labels)
  
  if(rescale){
    mutualInfoCompactForPosterior <- rescale(mutualInfoCompactForPosterior)
    rfErrors <- rescale(rfErrors)
  }
  
  miWithClassificationError <- data.frame(range, mutualInfoCompactForPosterior,
                                          rfErrors)
  
  miWithClassificationErrorsMelted <- melt(data = miWithClassificationError, 
                                           id.vars = "range")
  
  dev.new()
  ggplot(data = miWithClassificationErrorsMelted, aes(x = range, y = value, 
                                                      color = factor(variable, labels = c("Mutual Information", "Random Forest error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)+
    ggtitle(header)
  
}

plotFourMethods <- function(ldaTuningRange, ldaTuningObject){
  
  ldaTuningResults <- ldaTuningObject$ldatuningResults
  
  gryffith <- rescale(ldaTuningResults$Griffiths2004)
  cao <- rescale(ldaTuningResults$CaoJuan2009)
  arun <- rescale(ldaTuningResults$Arun2010)
  deveaud <- rescale(ldaTuningResults$Deveaud2014)

  dfGgPlot <- data.frame(ldaTuningRange, gryffith, cao, arun, deveaud)
  dfGgPlotMelted <- melt(data = dfGgPlot, id.vars = "ldaTuningRange")
  
  dev.new()
  ggplot(data = dfGgPlotMelted, aes(x = ldaTuningRange, y = value, color = factor(variable, 
    labels = c("Likelihood",  "Cosine similarity", "Arun", "Deveaud")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Methods") + theme_classic(base_size = 18)
}


library(scales)
library(reshape2)
library(ggplot2)

######################################### Estimate topic count ENTROPY 300 ##############################################

appsLdaModels <- extendedModels
appsRange <- topics_number
appsRfErrors <- c(accuracies_200,moreAccuracies)

appsRange_300 <- appsRange[1:24]

maxEntropyAppData <- sapply(appsRange_300, log)
maxEntropyAppDataRescaled = rescale(maxEntropyAppData)
appsAvgEntropyForModels_300 <- lapply(appsLdaModels[1:24], avgEntropyForModel)
appsMaxEntropyForModels_300 <- rescale(unlist(lapply(appsLdaModels[1:24], maxEntropyForModel)))
appsMinEntropyForModels_300 <- rescale(unlist(lapply(appsLdaModels[1:24], minEntropyForModel)))

appsAvgEntropyForModelsRescaled_300 <- rescale(unlist(appsAvgEntropyForModels_300))

appsAvgEntropyForPosterior_300 <- lapply(appsLdaModels[1:24], avgEntropyForPosterior1)
appsAvgEntropyForPosteriorRescaled_300 <- rescale(unlist(appsAvgEntropyForPosterior_300))

appsRfErrorsVector_300 <- unlist(appsRfErrors[1:24])
appsRfErrorsVectorRescaled_300 <- rescale(appsRfErrorsVector_300)

appsAvgEntropyForModelsRescaled_300
appsAvgEntropyForPosteriorRescaled_300
appsRfErrorsVectorRescaled_300

appsEntropyWithRfErrors_300 <- data.frame(appsRange_300, appsAvgEntropyForModelsRescaled_300, 
                                          appsAvgEntropyForPosteriorRescaled_300, appsMaxEntropyForModels_300, appsMinEntropyForModels_300,
                                          maxEntropyAppDataRescaled,
                                          appsRfErrorsVectorRescaled_300)

appsEntropyWithRfErrorsMelted_300 <- melt(data = appsEntropyWithRfErrors_300, id.vars = "appsRange_300")

dev.new()
ggplot(data = appsEntropyWithRfErrorsMelted_300, aes(x = appsRange_300, y = value, 
  color = factor(variable, labels = c("Average entropy of topics", "Average entropy of topics proportions", "Max topics entropy", "Min topics entropy", "Max entropy", "Random Forest error")))) + 
  geom_point() + geom_line(size=1) + labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)

######################################### MAX ENTROPY with posterior ##############################################

maxEntropyAppData <- sapply(appsRange_300, log)
posteriorEntropyAppData <- unlist(appsAvgEntropyForPosterior_300)
maxEntropyWithPosteriorAppData <- data.frame(appsRange_300,  
                                      posteriorEntropyAppData, 
                                      maxEntropyAppData, appsRfErrorsVectorRescaled_300)
maxEntropyWithPosteriorAppDataMelted <- melt(data = maxEntropyWithPosteriorAppData, id.vars = "appsRange_300")
dev.new()
ggplot(data = maxEntropyWithPosteriorAppDataMelted, aes(x = appsRange_300, y = value, 
  color = factor(variable, labels = c("posterior entropy", "max entropy", error)))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)

######################################  posterior entropy small values #############################

appsRfErrorsVectorRescaled_40 <- appsRfErrorsVectorRescaled_300[1:7]

appsRange_40 <- appsRange[1:7]
maxEntropyAppData <- sapply(appsRange_40, log)
posteriorEntropyAppData <- unlist(appsAvgEntropyForPosterior_300)
maxEntropyWithPosteriorAppData <- data.frame(appsRange_40,  
                                             posteriorEntropyAppData[1:7], 
                                             maxEntropyAppData, appsRfErrorsVectorRescaled_40)
maxEntropyWithPosteriorAppDataMelted <- melt(data = maxEntropyWithPosteriorAppData, id.vars = "appsRange_40")
dev.new()
ggplot(data = maxEntropyWithPosteriorAppDataMelted, aes(x = appsRange_40, y = value, 
  color = factor(variable, labels = c("posterior entropy", "max entropy", "rf error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)


###################################### Analyse posterior #############################

appsLdaModels[[1]]$topicmodel@alpha
dim(appsLdaModels[[2]]$topicmodel@beta)

appsLdaModels[[1]]$topicmodel@iter

iters_all <- sapply(appsLdaModels, function(m){
  m$topicmodel@iter
})
iters_all

alphas_all <- sapply(appsLdaModels, function(m){
  m$topicmodel@alpha
})
alphas_all


plot(appsRange[1:7], alphas_all[1:7], type="l")

reviewsRange[13:28]

iters_all

plot(appsRange[1:24], iters_all[1:24], type="l")

#################################################  alpha vs entropy ####################

apps_alphas_all <- sapply(appsLdaModels, function(m){
  m$topicmodel@alpha
})

apps_entropy_all <- sapply(appsLdaModels, function(m){
  avgEntropyForPosterior1(m)
})

apps_alphas_100_rescaled <- rescale(apps_alphas_all[1:13])
apps_entropy_100_rescaled <- rescale(apps_entropy_all[1:13])
appsRange_100 <- appsRange[1:13]

apps_alphaVsEntropy_100 <- data.frame(appsRange_100, apps_alphas_100_rescaled, apps_entropy_100_rescaled)
apps_alphaVsEntropyMelted_100 <- melt(data = apps_alphaVsEntropy_100, id.vars = "appsRange_100")

dev.new()
ggplot(data = apps_alphaVsEntropyMelted_100, aes(x = appsRange_100, y = value, color = factor(variable, 
  labels = c("Alphas",  "Entropies")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Methods") + theme_classic(base_size = 18)

################### differential entropy of dirichlet ###############################

multLogBeta <- function(alpha, dim){
  left <- dim*lgamma(alpha)
  right <- lgamma(dim*alpha)
  left - right
}

dirEntropy <- function(alpha, dim){
  multLogBeta(alpha, dim) + (dim*alpha - dim)*(digamma(dim*alpha) - digamma(alpha))
}

dirEntropy(alpha = 137, dim=15)
multLogBeta(alpha = 1, dim=15)

############################## analyse posterior ###############################

analysePosteriorEntropy(appsRange[1:24], appsLdaModels[1:24], header="Mobile Apps")

############### analyses of average entropy ###########################
subrange = 1:28
analyseAverageEntropyNoSvm(appsRange[subrange], appsLdaModels[subrange], appsRfErrors[subrange], header="Mobile Apps")


######################################### Estimate topic count conditional entropy ENTROPY ##############################################

appsTrainLabels <- trainLabels
appsTestLabels <- testLabels
appsRfErrors <- c(accuracies_200,moreAccuracies)

subrange = 2:28

analyseJoinedEntropyNoSvm(appsRange[subrange], appsLdaModels[subrange], appsTrainLabels, appsRfErrors[subrange], 
                     header="Mobile Apps, bins=2", bins=2)

analyseJoinedEntropyNoSvm(appsRange[subrange], appsLdaModels[subrange], appsTrainLabels, appsRfErrors[subrange], 
                          header="Mobile Apps, bins=10", bins=10)

analyseConditionalEntropyNoSvm(appsRange[subrange], appsLdaModels[subrange], appsTrainLabels, appsRfErrors[subrange], 
                               header="Mobile Apps, bins=25", bins=25, rescale = TRUE, runOnTest = FALSE)

######################################### predict using ensemble ##############################################

subRange <- c(4, 5, 6, 8, 9, 10,11,12,13)
accuracies_200[subRange]
subRfModels_200 <- rfModels_200[subRange]
subLdaModels_200 <- lda_models_200[subRange]

appsRfModels <- c(rfModels_200, rfModels_300_1000)

length(appsRfModels)
length(appsLdaModels)

res <- predictEnsemble(appsRfModels[subRange], appsLdaModels[subRange])
errorForEnsembleResult(res, testLabels)

res <- predictWithScoreEnsemble(appsRfModels[subRange], appsLdaModels[subRange])
errorForEnsembleWithScoreResult(res$classes, res$scores, testLabels)

#analyseEnsemble(appsRfModels[subRange], appsLdaModels[subRange], testLabels)

chosenModels <- chooseModelUsingPerplexity(tfidfData$cleanedTestMatrix, appsLdaModels)
chosenModels

errorPerplexityEnsemble(chosenModels, appsLdaModels, appsRfModels)

######################################### Analyse mutual information #########################################

appsSubrange=1:10
analyseMutualInformationNoSvm(appsRange[appsSubrange], appsLdaModels[appsSubrange], appsTrainLabels, appsRfErrors[appsSubrange], 
                         header="Mobile Apps, bins=25", bins=25, rescale = TRUE)

######################################### Compare models using KL between original and sampled data #########################################

appsSubrangeForSampling <- 1:28

mobileAppsKlDataVsSample <- calculateKlBetweenDataAndSample(tfidfData$cleanedTrainMatrix, appsLdaModels[appsSubrangeForSampling])

dev.new()
plot(appsRange[appsSubrangeForSampling], mobileAppsKlDataVsSample, type="l")

######################################### Compare models using per-class KL between original and sampled data #########################################

appsSubrangeForSampling <- 1:28

originalTrainMatrix <- as.matrix(tfidfData$cleanedTrainMatrix)

docLengths <- rowSums(originalTrainMatrix)
appsSamplesFromLdaModels <- lapply(appsLdaModels[appsSubrangeForSampling], function(m){
  generateCorpusFromModel(m, docLengths)
})

klAvgs <- calculateAvgKl(range = appsRange[appsSubrangeForSampling], samplesFromModel = appsSamplesFromLdaModels,
               labels = appsTrainLabels, tfMatrix = tfidfData$cleanedTrainMatrix)
dev.new()
plot(appsRange[appsSubrangeForSampling], klAvgs, type="l")



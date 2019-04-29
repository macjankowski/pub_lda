library(readtext)
library(entropy)    
library (lda)
library(MASS)
library(scales)
library(ggplot2)
library(reshape2)
library(rpart)
library(e1071)
library(naivebayes)




source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')
source('./ensemble.R')
source('./charts.R')
source('./multTest.R')
source('./sampleFromModel.R')

tree_number <- 500

filePath = '/Users/mjankowski/doc/data/mix20_rand700_tokens_0211/tokens'
negativeReviewsPath = '/Users/mjankowski/doc/data/mix20_rand700_tokens_0211/tokens/neg'
negativeReviewsPath
positiveReviewsPath = '/Users/mjankowski/doc/data/mix20_rand700_tokens_0211/tokens/pos'
positiveReviewsPath

negativeReviews <- readtext(negativeReviewsPath, docvarsfrom = "filenames", 
                docvarnames = c("prefix", "label"), dvsep = "-")

negativeReviews$label <- 0
negativeReviews

positiveReviews <- readtext(positiveReviewsPath, docvarsfrom = "filenames", 
                            docvarnames = c("prefix", "label"), dvsep = "-")
positiveReviews$label <- 1
positiveReviews

dim(positiveReviews)
dim(negativeReviews)

allReviewsDfUnshuffled <- rbind(positiveReviews, negativeReviews)

allReviewsDfUnshuffled[1:20,]$label

dim(allReviewsDfUnshuffled)
allReviewsDfAllColumns <- allReviewsDfUnshuffled[sample(nrow(allReviewsDfUnshuffled)),]
allReviewsDf <- subset(allReviewsDfAllColumns, select=c("label", "text"))

allReviewsDf[1:20,]$label
dim(allReviewsDf)

partitionedReviews <- partitionData(allReviewsDf)
dim(partitionedReviews$train)
dim(partitionedReviews$test)

partitionedReviewsTF <- prepareTfIdfWithLabels(partitionedReviews, sparseLevel=0.95)

partitionedReviewsTF$cleanedTestLabels

dim(partitionedReviewsTF$cleanedTrainMatrix)
length(partitionedReviewsTF$cleanedTrainLabels)
dim(partitionedReviewsTF$cleanedTestMatrix)
length(partitionedReviewsTF$cleanedTestLabels)
tree_number

reviewsTrainLabels <- partitionedReviewsTF$cleanedTrainLabels

reviewsRfModel <- randomForest(x=as.matrix(partitionedReviewsTF$cleanedTrainMatrix), y=reviewsTrainLabels, ntree=tree_number, keep.forest=TRUE)
reviewRsfModel

predictSimple(reviewRsfModel, partitionedReviewsTF$cleanedTestMatrix, partitionedReviewsTF$cleanedTestLabels)

reviewsSVMModel <- svm(x=as.matrix(partitionedReviewsTF$cleanedTrainMatrix), y=reviewsTrainLabels)
reviewsSVMModel

predictSimple(reviewsSVMModel, partitionedReviewsTF$cleanedTestMatrix, partitionedReviewsTF$cleanedTestLabels)


# naive bayes

reviewsNbModel <- naive_bayes(x=as.matrix(partitionedReviewsTF$cleanedTrainMatrix), y=reviewsTrainLabels, laplace = 0.0001)
reviewsNbModel

predictSimple(reviewsNbModel, partitionedReviewsTF$cleanedTestMatrix, partitionedReviewsTF$cleanedTestLabels)

# linear discriminant analysis
reviewsLinearDAModel <- lda(x=as.matrix(partitionedReviewsTF$cleanedTrainMatrix), grouping = reviewsTrainLabels)
reviewsLinearDAModel

predictSimpleLinearDiscriminantAnalysis(reviewsLinearDAModel, as.matrix(partitionedReviewsTF$cleanedTestMatrix), partitionedReviewsTF$cleanedTestLabels)

reviewsGlmModel <- glm.fit(x=as.matrix(partitionedReviewsTF$cleanedTrainMatrix), y = reviewsTrainLabels)
reviewsGlmModel

######################################### experiment up to 1000 topics ##############################################

reviewsRange <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000)
reviewsLdaModels <- lapply(reviewsRange, function(x){calculateLDA(tfData = partitionedReviewsTF, topic_number = x)})

reviewsTrainLabels <- partitionedReviewsTF$cleanedTrainLabels
reviewsTestLabels <- partitionedReviewsTF$cleanedTestLabels

reviewsRfModels <- lapply(reviewsLdaModels, function(x){
  trainRfOnLda(x, reviewsTrainLabels)
})

reviewsRfErrors <- sapply(1:length(reviewsRange), function(i){
  predictSimple(reviewsRfModels[[i]], reviewsLdaModels[[i]]$ldaTestData, reviewsTestLabels)
})

reviewsSVMModels <- lapply(reviewsLdaModels, function(x){
  trainSVMOnLda(x, reviewsTrainLabels)
})

reviewsSVMErrors <- sapply(1:length(reviewsRange), function(i){
  predictSimple(reviewsSVMModels[[i]], reviewsLdaModels[[i]]$ldaTestData, reviewsTestLabels)
})

reviewsNbModels <- lapply(reviewsLdaModels, function(x){
  trainNbOnLda(x, reviewsTrainLabels)
})

reviewsNbErrors <- sapply(1:length(reviewsRange), function(i){
  predictSimple(reviewsNbModels[[i]], reviewsLdaModels[[i]]$ldaTestData, reviewsTestLabels)
})

reviewsLinearDiscriminantAnalysisModels <- lapply(reviewsLdaModels, function(x){
  trainLinearDiscriminantAnalysisOnLda(x, reviewsTrainLabels)
})

reviewsLinearDiscriminantAnalysisErrors <- sapply(1:length(reviewsRange), function(i){
  predictSimpleLinearDiscriminantAnalysis(reviewsLinearDiscriminantAnalysisModels[[i]], reviewsLdaModels[[i]]$ldaTestData, reviewsTestLabels)
})

plot(reviewsRange, reviewsSVMErrors, type="l")
plot(reviewsRange, reviewsRfErrors, type="l")
plot(reviewsRange, reviewsNbErrors, type="l")


######################################### Estimate topic count ENTROPY ##############################################
length(reviewsLdaModels)

reviewsAvgEntropyForModels <- lapply(reviewsLdaModels, avgEntropyForModel)
reviewsAvgEntropyForModelsRescaled <- rescale(unlist(reviewsAvgEntropyForModels))
reviewsAvgEntropyForModelsRescaled

reviewsRfErrorsVector <- unlist(reviewsRfErrors)

reviewsRfErrorsVectorRescaled <- rescale(reviewsRfErrorsVector)

reviewsEntropyWithRfErrors <- data.frame(reviewsRange, reviewsAvgEntropyForModelsRescaled, reviewsRfErrorsVectorRescaled)
reviewsEntropyWithRfErrorsMelted <- melt(data = reviewsEntropyWithRfErrors, id.vars = "reviewsRange")

dev.new()
ggplot(data = reviewsEntropyWithRfErrorsMelted, aes(x = reviewsRange, y = value, 
                           color = factor(variable, labels = c("Average entropy of topics",  "Random Forest error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)



######################################### Estimate topic count ENTROPY inne ##############################################
length(reviewsLdaModels)

reviewsAvgEntropyForModels <- lapply(reviewsLdaModels, avgEntropyForModel)
reviewsMaxEntropyForModels <- rescale(unlist(lapply(reviewsLdaModels, maxEntropyForModel)))
reviewsMinEntropyForModels <- rescale(unlist(lapply(reviewsLdaModels, minEntropyForModel)))

reviewsAvgEntropyForModelsRescaled <- rescale(unlist(reviewsAvgEntropyForModels))

reviewsAvgEntropyForPosterior <- lapply(reviewsLdaModels, avgEntropyForPosterior1)
reviewsAvgEntropyForPosteriorRescaled <- rescale(unlist(reviewsAvgEntropyForPosterior))

reviewsRfErrorsVector <- unlist(reviewsRfErrors)
reviewsRfErrorsVectorRescaled <- rescale(reviewsRfErrorsVector)


reviewsAvgEntropyForModelsRescaled
reviewsAvgEntropyForPosteriorRescaled
reviewsRfErrorsVectorRescaled


reviewsEntropyWithRfErrors <- data.frame(reviewsRange, reviewsAvgEntropyForModelsRescaled, reviewsAvgEntropyForPosteriorRescaled, reviewsMaxEntropyForModels, reviewsMinEntropyForModels, reviewsRfErrorsVectorRescaled)
reviewsEntropyWithRfErrorsMelted <- melt(data = reviewsEntropyWithRfErrors, id.vars = "reviewsRange")

dev.new()
ggplot(data = reviewsEntropyWithRfErrorsMelted, aes(x = reviewsRange, y = value, 
  color = factor(variable, labels = c("Average entropy of topics", "Average entropy of topics proportions", "Max entropy", "Min entropy", "Random Forest error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)

######################################### Estimate topic count ENTROPY 300 ##############################################
length(reviewsLdaModels)

reviewsRange_300 <- reviewsRange[1:24]

reviewsAvgEntropyForModels_300 <- lapply(reviewsLdaModels[1:24], avgEntropyForModel)
reviewsMaxEntropyForModels_300 <- rescale(unlist(lapply(reviewsLdaModels[1:24], maxEntropyForModel)))
reviewsMinEntropyForModels_300 <- rescale(unlist(lapply(reviewsLdaModels[1:24], minEntropyForModel)))

reviewsAvgEntropyForModelsRescaled_300 <- rescale(unlist(reviewsAvgEntropyForModels_300))

reviewsAvgEntropyForPosterior_300 <- lapply(reviewsLdaModels[1:24], avgEntropyForPosterior1)
reviewsAvgEntropyForPosteriorRescaled_300 <- rescale(unlist(reviewsAvgEntropyForPosterior_300))

reviewsRfErrorsVector_300 <- unlist(reviewsRfErrors[1:24])
reviewsRfErrorsVectorRescaled_300 <- rescale(reviewsRfErrorsVector_300)

reviewsSVMErrorsVector_300 <- unlist(reviewsSVMErrors[1:24])
reviewsSVMErrorsVectorRescaled_300 <- rescale(reviewsSVMErrorsVector_300)


reviewsAvgEntropyForModelsRescaled_300
reviewsAvgEntropyForPosteriorRescaled_300
reviewsRfErrorsVectorRescaled_300


reviewsEntropyWithRfErrors_300 <- data.frame(reviewsRange_300, reviewsAvgEntropyForModelsRescaled_300, 
                                             reviewsAvgEntropyForPosteriorRescaled_300, reviewsMaxEntropyForModels_300, 
                                             reviewsMinEntropyForModels_300, reviewsRfErrorsVectorRescaled_300,
                                             reviewsSVMErrorsVectorRescaled_300, rescale(reviewsPerplexities[1:24]))
reviewsEntropyWithRfErrorsMelted_300 <- melt(data = reviewsEntropyWithRfErrors_300, id.vars = "reviewsRange_300")

dev.new()
ggplot(data = reviewsEntropyWithRfErrorsMelted_300, aes(x = reviewsRange_300, y = value, 
  color = factor(variable, labels = c("Average entropy of topics", "Average entropy of topics proportions", 
                                      "Max entropy", "Min entropy", "Random Forest error", "SVM Error", "Perplexities")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)


######################################### Estimate topic count ENTROPY best charts only ##############################################

reviewsEntropyWithClassificationErrors_300 <- data.frame(reviewsRange_300, 
                                             reviewsMinEntropyForModels_300, reviewsRfErrorsVectorRescaled_300,
                                             reviewsSVMErrorsVectorRescaled_300)
reviewsEntropyWithClassificationErrorsMelted_300 <- melt(data = reviewsEntropyWithClassificationErrors_300, id.vars = "reviewsRange_300")

dev.new()
ggplot(data = reviewsEntropyWithClassificationErrorsMelted_300, aes(x = reviewsRange_300, y = value, 
                                                        color = factor(variable, labels = c("Min entropy", "Random Forest error", "SVM Error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)


######################################### MAX ENTROPY with posterior ##############################################

maxEntropy <- sapply(reviewsRange_300, log)
posteriorEntropy <- unlist(reviewsAvgEntropyForPosterior_300)
maxEntropyWithPosterior <- data.frame(reviewsRange_300,  
                                      posteriorEntropy, 
                                      maxEntropy)
maxEntropyWithPosteriorMelted <- melt(data = maxEntropyWithPosterior, id.vars = "reviewsRange_300")
dev.new()
ggplot(data = maxEntropyWithPosteriorMelted, aes(x = reviewsRange_300, y = value, 
  color = factor(variable, labels = c("posterior entropy", "max entropy")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)

####################################### movie reviews ggplot 2-200 topic number scores with accuracies #############################################

reviewsLdaTuning <- estimateTopicsCount4MethodsRange(reviewsRange, tfidfData = partitionedReviewsTF)

reviewsLdaTuningResults <- reviewsLdaTuning$ldatuningResults

reviewsGryffith <- rescale(reviewsLdaTuningResults$Griffiths2004)
reviewsCao <- rescale(reviewsLdaTuningResults$CaoJuan2009)
reviewsArun <- rescale(reviewsLdaTuningResults$Arun2010)
reviewsDeveaud <- rescale(reviewsLdaTuningResults$Deveaud2014)
reviewsRfErrorsRescaled <- rescale(reviewsRfErrors)

reviewsDfggplot <- data.frame(reviewsRange, reviewsGryffith, reviewsCao, reviewsArun, reviewsDeveaud, reviewsRfErrorsRescaled, reviewsAvgEntropyForModelsRescaled)
reviewsDfggplotMelted <- melt(data = reviewsDfggplot, id.vars = "reviewsRange")

dev.new()
ggplot(data = reviewsDfggplotMelted, aes(x = reviewsRange, y = value, color = factor(variable, 
  labels = c("Likelihood",  "Cosine similarity", "Arun", "Deveaud", "Random Forest error", "Average Entropy")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Methods") + theme_classic(base_size = 18)


############################################## perplexities ########################################

length(reviewsLdaModels)
reviewsPerplexities <- sapply(reviewsLdaModels, function(x){ perplexity(x$topicmodel, partitionedReviewsTF$cleanedTestMatrix)})

plot(reviewsRange, reviewsPerplexities, type="l")

####################################### mutual information #############################################
library(arules)

reviewsMi <- avg_mi_all_models(reviewsLdaModels, reviewsTrainLabels)
plot(reviewsRange, reviewsMi, type="l")

all_models_avg_spearman <- avg_spearman_all_models(lda_models_200, trainLabels)
plot(range_200, all_models_avg_spearman, type="l")

all_models_avg_spearman

avg_spearman_single_model_without_mean(lda_models_200[[2]]$ldaTrainData, (1:length(trainLabels)))

max(abs(harmonic.mean(avg_spearman_single_model_without_mean(lda_models_200[[2]]$ldaTrainData, trainLabels))))

###################################### Analyse posterior #############################

reviewsRange_300[14]
reviewsLdaModels
posterior_tmp <- posterior(reviewsLdaModels[[3]]$topicmodel)
dim(posterior_tmp$topics)
length(posterior_tmp$topics[1,])

apply(posterior_tmp$topics,1,entropy.plugin)
log(15)

entropy.plugin(posterior_tmp$topics[100,])
log(110)

reviewsLdaModels[[1]]$topicmodel@alpha


dim(reviewsLdaModels[[2]]$topicmodel@beta)

reviewsLdaModels[[1]]$topicmodel@iter

iters_all <- sapply(reviewsLdaModels, function(m){
  m$topicmodel@iter
})

alphas_all <- sapply(reviewsLdaModels, function(m){
  m$topicmodel@alpha
})

entropy_all <- sapply(reviewsLdaModels, function(m){
  avgEntropyForPosterior1(m)
})

alphaVsEntropy <- data.frame(reviewsRange, rescale(alphas_all), rescale(entropy_all))
alphaVsEntropyMelted <- melt(data = alphaVsEntropy, id.vars = "reviewsRange")

dev.new()
ggplot(data = alphaVsEntropyMelted, aes(x = reviewsRange, y = value, color = factor(variable, 
  labels = c("Alphas",  "Entropies")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Methods") + theme_classic(base_size = 18)


######## 2 topics ###################
posterior_tmp <- posterior(reviewsLdaModels[[1]]$topicmodel)
reviewsLdaModels[[1]]$topicmodel@alpha

posterior_tmp$topics[100,]
entropy.plugin(posterior_tmp$topics[100,])
log(2)

######################################### experiment fixed alpha 0-100 ##############################################
reviewsRange_noAlpha_100 <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100)
reviewsLdaModels_noAlpha_100  <- lapply(reviewsRange_noAlpha_100, function(x){calculateLDA(tfData = partitionedReviewsTF, topic_number = x, estimateAlpha = FALSE)})


######################################### experiment fixed alpha 100-200 ##############################################

reviewsRange_noAlpha <- c(110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
reviewsLdaModels_noAlpha <- lapply(reviewsRange_noAlpha, function(x){calculateLDA(tfData = partitionedReviewsTF, topic_number = x, estimateAlpha = FALSE)})

reviewsRfModels_noAlpha <- lapply(reviewsLdaModels_noAlpha, function(x){
  trainRfOnLda(x, reviewsTrainLabels)
})

reviewsRfErrors_noAlpha <- sapply(1:length(reviewsRange_noAlpha), function(i){
  predictSimple(reviewsRfModels_noAlpha[[i]], reviewsLdaModels_noAlpha[[i]]$ldaTestData, reviewsTestLabels)
})

reviewsRfErrors_100_200 <- reviewsRfErrors[14:23]

reviewsRfErrors_data_frame_100_200 <- data.frame(reviewsRange_noAlpha, reviewsRfErrors_100_200, reviewsRfErrors_noAlpha)
reviewsRfErrors_data_frame_100_200Melted <- melt(data = reviewsRfErrors_data_frame_100_200, id.vars = "reviewsRange_noAlpha")

dev.new()
ggplot(data = reviewsRfErrors_data_frame_100_200Melted, aes(x = reviewsRange_noAlpha, y = value, color = factor(variable, 
  labels = c("fitted alpha",  "const alpha")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Methods") + theme_classic(base_size = 18)


alphas_no_fitting <- sapply(reviewsLdaModels_noAlpha, function(m){
  m$topicmodel@alpha
})

alphas_no_fitting

entropy_no_fitting <- sapply(reviewsLdaModels_noAlpha, function(m){
  avgEntropyForPosterior1(m)
})

entropy_no_fitting

############################  model checking ###########################################

trainMatrix <- as.matrix(partitionedReviewsTF$cleanedTrainMatrix)
dim(trainMatrix)
# first document
trainMatrix[1,]

# first document from first model
posterior_tmp_1 <- posterior(reviewsLdaModels[[1]]$topicmodel)
dim(posterior_tmp_1$topics)
posterior_tmp_1$topics[1,]
dim(posterior_tmp_1$terms)

posterior_tmp_1$terms[1,1:10]

######################################### Estimate topic count max posterior ENTROPY ##############################################

reviewsRfErrors_noAlpha_add_zeroes <- c(rep(0,13), reviewsRfErrors_noAlpha,0)

reviewsGryffith <- rescale(reviewsLdaTuningResults$Griffiths2004)

reviewsPosteriorEntropyWithClassificationErrors_300 <- data.frame(reviewsRange_300,  
                                                                  reviewsGryffith[1:24],
                                                                  reviewsAvgEntropyForPosteriorRescaled_300, 
                                                                  reviewsRfErrorsVectorRescaled_300,
                                                                  reviewsSVMErrorsVectorRescaled_300, 
                                                                  reviewsRfErrors_noAlpha_add_zeroes)
reviewsPosteriorEntropyWithClassificationErrorsMelted_300 <- melt(data = reviewsPosteriorEntropyWithClassificationErrors_300, id.vars = "reviewsRange_300")

dev.new()
ggplot(data = reviewsPosteriorEntropyWithClassificationErrorsMelted_300, aes(x = reviewsRange_300, y = value, 
  color = factor(variable, labels = c("Likelihood", "Average entropy of topics proportions", "Random Forest error", "SVM Error","Random Forest error (no alpha fitting)")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)

#################################### analyse posterior ###################################

analysePosteriorEntropy(reviewsRange_300, reviewsLdaModels[1:24], "Moview Reviews")

reviewsLdaModels[[3]]$topicmodel@loglikelihood
reviewsLdaModels[[3]]$topicmodel@beta
exp(reviewsLdaModels[[3]]$topicmodel@beta[1:10,1:10])

posterior <- posterior(reviewsLdaModels[[3]]$topicmodel)
reviewsLdaModels[[3]]$topicmodel@beta

dim(posterior$topics)
dim(reviewsLdaModels[[3]]$topicmodel@beta)

prob_m <- posterior$topics %*% exp(reviewsLdaModels[[3]]$topicmodel@beta)
dim(prob_m)
rowSums(prob_m)


######################################### Average entropy ##############################################
subrange = 1:28
analyseAverageEntropy(reviewsRange[subrange], reviewsLdaModels[subrange], reviewsRfErrors[subrange], 
                      reviewsSVMErrors[subrange], rescale=TRUE, header="Movie Reviews")

######################################### Estimate topic count conditional entropy ENTROPY ##############################################

infotheo::entropy(reviewsTrainLabels,method="emp")

movieRevieSubrange = 1:24
analyseJoinedEntropy(reviewsRange[movieRevieSubrange], reviewsLdaModels[movieRevieSubrange], reviewsTrainLabels, reviewsRfErrors[movieRevieSubrange], 
                     reviewsSVMErrors[movieRevieSubrange], header="Movie Reviews, bins=2", bins=2)

analyseConditionalEntropy(reviewsRange[movieRevieSubrange], reviewsLdaModels[movieRevieSubrange], as.numeric(reviewsTrainLabels), reviewsRfErrors[movieRevieSubrange], 
                     reviewsSVMErrors[movieRevieSubrange], header="Movie Reviews, bins=2", bins=2)

dev.new()
analyseMutualInformation(reviewsRange[movieRevieSubrange], reviewsLdaModels[movieRevieSubrange], as.numeric(reviewsTrainLabels), reviewsRfErrors[movieRevieSubrange], 
                          reviewsSVMErrors[movieRevieSubrange], header="Movie Reviews, bins=2", bins=2, rescale=TRUE)

conditionalEntropy2(reviewsLdaModels[[5]], reviewsTrainLabels, bins=2)

as.numeric(reviewsTrainLabels)



ttt <- reviewsLdaModels[[28]]$ldaTrainData
nRows <- dim(ttt)[1]
nCols <- dim(ttt)[2]
rawDisc <- arules::discretize(ttt, method = "frequency", categories = 2, onlycuts=FALSE)
trainDisc <- matrix(as.numeric(rawDisc), nrow = nRows, ncol=nCols)
joinedData <- data.frame(as.numeric(reviewsTrainLabels), trainDisc)
joinedData
joinedEntr <- infotheo::entropy(joinedData,method="emp") 
dataentr <-  infotheo::entropy(trainDisc, method = "emp")
joinedEntr
dataentr

H = joinedEntr - dataentr
H

############################ ensemble #############################

reviewsLdaModels[[1]]$topicmodel

partitionedReviewsTF$cleanedTestMatrix[4,]

perplexity(reviewsLdaModels[[1]]$topicmodel, partitionedReviewsTF$cleanedTestMatrix[4,])


subrange=7:13
movieReviewsChosenModels <- chooseModelUsingPerplexity(partitionedReviewsTF$cleanedTestMatrix, reviewsLdaModels[subrange])  
cbind(1:length(reviewsRfErrors), reviewsRange, reviewsRfErrors)

reviewsLdaModels[[13]]$topicmodel@k
betaDistribution <- exp(reviewsLdaModels[[13]]$topicmodel@beta)
dim(betaDistribution)
delta <- diri.est(betaDistribution[,1:500])
delta

######################################### Compare models using KL between original and sampled data #########################################

reviewsSubrangeForSampling <- 1:28
reviewsKlDataVsSample <- calculateKlBetweenDataAndSample(partitionedReviewsTF$cleanedTrainMatrix, reviewsLdaModels[reviewsSubrangeForSampling])

dev.new()
plot(reviewsRange[reviewsSubrangeForSampling], reviewsKlDataVsSample, type="l")

######################################### Compare models using per-class KL between original and sampled data #########################################

ssss <- generateCorpusFromModel(reviewsLdaModels[[2]], reviewsDocLengths)

reviewsTrainMatrix <- as.matrix(partitionedReviewsTF$cleanedTrainMatrix)

reviewsDocLengths <- rowSums(reviewsTrainMatrix)
reviewsSamplesFromLdaModels <- lapply(reviewsLdaModels[reviewsSubrangeForSampling], function(m){
  generateCorpusFromModel(m, reviewsDocLengths)
})

reviewsKlAvgs <- calculateAvgKl(range = reviewsRange[reviewsSubrangeForSampling], samplesFromModel = reviewsSamplesFromLdaModels,
                         labels = reviewsTrainLabels, tfMatrix = partitionedReviewsTF$cleanedTrainMatrix)
dev.new()
plot(reviewsRange[reviewsSubrangeForSampling], reviewsKlAvgs, type="l")

######################################### MultinomialTest #########################################

reviewsSubrangeForSampling <- 1:28
reviewsKlDataVsSample <- calculateKlBetweenDataAndSample(partitionedReviewsTF$cleanedTrainMatrix, reviewsLdaModels[reviewsSubrangeForSampling])

reviewMultTest <- aaa(partitionedReviewsTF$cleanedTrainMatrix, reviewsSamplesFromLdaModels[[6]])
reviewMultTest

######################################### Chi squared test #######################################

reviewsOriginalTrainMatrix <- as.matrix(partitionedReviewsTF$cleanedTrainMatrix)
reviewsDataMultinomialDistribution <- estimateMultinomialFromCorpus(reviewsOriginalTrainMatrix)

############### chi-sq with itself - obviously passed ######################

obsNult <- colSums(reviewsOriginalTrainMatrix)
t <- chisq.test(obsNult, p = reviewsDataMultinomialDistribution)
t$p.value

############## chi-sq with subset of itself - passed as expected ########################
shuffled <- sample(1:1108, 500, replace = FALSE)
obsNultSubset <- colSums(reviewsOriginalTrainMatrix[shuffled, ])
t <- chisq.test(obsNultSubset, p = reviewsDataMultinomialDistribution)
t$p.value

############# chi-sq with subset pertaining to one of the classes ##############################

subsetForClass_1 <- which(reviewsTrainLabels==1)
obsMultSubsetForClass_1 <- colSums(reviewsOriginalTrainMatrix[subsetForClass_1, ])
t <- chisq.test(obsMultSubsetForClass_1, p = reviewsDataMultinomialDistribution)
t$p.value
t$statistic

subsetForClass_0 <- which(reviewsTrainLabels==0)
reviewsDataMultinomialDistributionClass_0 <- estimateMultinomialFromCorpus(reviewsOriginalTrainMatrix[subsetForClass_0, ])

subsetForClass_1 <- which(reviewsTrainLabels==1)
obsMultSubsetForClass_1 <- colSums(reviewsOriginalTrainMatrix[subsetForClass_1, ])
t <- chisq.test(obsMultSubsetForClass_1, p = reviewsDataMultinomialDistributionClass_0)
t$p.value
t$statistic


###########################################################

dim(reviewsOriginalTrainMatrix)

sample1ObservedCounts <- observedCountsFromCorpus(reviewsSamplesFromLdaModels[[13]])

t <- chisq.test(sample1ObservedCounts, p = reviewsDataMultinomialDistribution)

t$statistic
t$p.value
t$parameter

reviewsSubrangeForSampling <- 1:28
reviewsPValueDataVsSample <- calculatePValueBetweenDataAndSample(
  partitionedReviewsTF$cleanedTrainMatrix, 
  reviewsLdaModels[reviewsSubrangeForSampling],
  reviewsSamplesFromLdaModels
)

chi = calculatePValueBetweenDataAndSampleForClass(
  partitionedReviewsTF$cleanedTrainMatrix, 
  reviewsLdaModels[reviewsSubrangeForSampling],
  reviewsSamplesFromLdaModels,
  labels = reviewsTrainLabels
)
sapply(chi, function(c){c$p.value})
sapply(chi, function(c){c$statistic})


dev.new()
plot(reviewsRange[reviewsSubrangeForSampling], reviewsPValueDataVsSample, type="l")

calculatePValuePerClassForModel(
  classIndex = 0, 
  labels = reviewsTrainLabels, 
  originalTrainTf = partitionedReviewsTF$cleanedTrainMatrix,
  sampleFromModel = reviewsSamplesFromLdaModels[[13]]
)


######################################### per class p-value ############################################

reviewsSubrangeForSampling<-1:28
reviewsPValueDataVsSampleSubset <- calculatePerClassPValueBetweenDataAndSample(
  originalTrainTf = partitionedReviewsTF$cleanedTrainMatrix,
  models = reviewsLdaModels[reviewsSubrangeForSampling],
  labels = reviewsTrainLabels,
  classLabel = 0
)
reviewsPValueDataVsSampleSubset[reviewsPValueDataVsSampleSubset > 0.05]
reviewsPValueDataVsSampleSubset

######################################### per class kl ############################################

reviewsSubrangeForSampling<-1:28
reviewsKlDataVsSampleSubset <- calculateKlPerClassForModel(
  classIndex = 0,
  labels = reviewsTrainLabels,
  originalTrainTf = partitionedReviewsTF$cleanedTrainMatrix,
  sampleFromModel = reviewsSamplesFromLdaModels[[13]]
)
#reviewsKlDataVsSampleSubset[reviewsKlDataVsSampleSubset > 0.05]
reviewsKlDataVsSampleSubset

###################################### per class likelihood #################################

log_likelihood <- modelPerClassLikelihood(
  labels = reviewsTrainLabels,
  originalTrainTf = partitionedReviewsTF$cleanedTrainMatrix,
  sampleFromModel = reviewsSamplesFromLdaModels[[11]]
)
log_likelihood

logLikelihoodsForModels <- lapply(reviewsSamplesFromLdaModels, function(sample){
  modelPerClassLikelihood(
    labels = reviewsTrainLabels,
    originalTrainTf = partitionedReviewsTF$cleanedTrainMatrix,
    sampleFromModel = sample
  )
})

plot(reviewsRange, logLikelihoodsForModels, type="l")
reviewsRange[which.min(unlist(logLikelihoodsForModels))]

logLikelihoodsForModels_onTest <- unlist(lapply(reviewsSamplesFromLdaModels, function(sample){
  modelPerClassLikelihood(
    labels = reviewsTestLabels,
    originalTrainTf = partitionedReviewsTF$cleanedTestMatrix,
    sampleFromModel = sample
  )
}))

plot(reviewsRange, logLikelihoodsForModels_onTest, type="l")
reviewsRange[order(logLikelihoodsForModels_onTest, decreasing = TRUE)]


reviewsRange[which.min(logLikelihoodsForModels_onTest[logLikelihoodsForModels_onTest != min(logLikelihoodsForModels_onTest)])]

n <- length(logLikelihoodsForModels_onTest)
sort(logLikelihoodsForModels_onTest,partial=n-1)[n-1]

################################## lik for model #########################

logLikelihoodsForModels_2 <- unlist(lapply(reviewsSamplesFromLdaModels, function(sample){
  calculateLikelihoodForModel(
    labels = reviewsTrainLabels,
    originalTrainTf = partitionedReviewsTF$cleanedTrainMatrix,
    sampleFromModel = sample
  )
}))
plot(reviewsRange, logLikelihoodsForModels_2, type="l")
reviewsRange[order(logLikelihoodsForModels_2, decreasing = FALSE)]

## on test ##############
logLikelihoodsForModels_onTest_2 <- unlist(lapply(reviewsSamplesFromLdaModels, function(sample){
  calculateLikelihoodForModel(
    labels = reviewsTestLabels,
    originalTrainTf = partitionedReviewsTF$cleanedTestMatrix,
    sampleFromModel = sample
  )
}))
plot(reviewsRange, logLikelihoodsForModels_onTest_2, type="l")
reviewsRange[order(logLikelihoodsForModels_onTest_2, decreasing = TRUE)]




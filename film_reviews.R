library(readtext)
library(entropy)    
library (lda)
library(MASS)
library(scales)
library(ggplot2)
library(reshape2)
library(rpart)
library(e1071)



source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')
source('./ensemble.R')

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

plot(reviewsRange, reviewsSVMErrors, type="l")
plot(reviewsRange, reviewsRfErrors, type="l")

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


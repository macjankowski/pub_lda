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

filePath = '/Users/mjankowski/doc/data/sms_spam/SMSSpamCollection'

smsSpamFile <- read.table(filePath, sep="\t", header=FALSE, quote = "")
dim(smsSpamFile)
smsLabelMapping <- data.frame(label = c("ham", "spam"), label = c(0,1))

smsSpamFile[1:10,]



allSmsSpamData <- merge(smsSpamFile, smsLabelMapping, by.x = "V1", by.y = "label")[,c(3,2)]
names(allSmsSpamData) <- c("label", "text")
dim(allSmsSpamData)

spam <- allSmsSpamData[allSmsSpamData$label == 1,]
nonSpam <- allSmsSpamData[allSmsSpamData$label == 0,]

dim(spam)
dim(nonSpam)

nonSpamSample <- nonSpam[sample(nrow(nonSpam), 747),]
dim(nonSpamSample)

smsSpamDataTmp <- rbind(spam, nonSpamSample)
dim(smsSpamDataTmp)

smsSpamData <- smsSpamDataTmp[sample(nrow(smsSpamDataTmp)),]
dim(smsSpamData)

smsSpamData[1:20,]


partitionedSmsSpam <- partitionData(smsSpamData)
dim(partitionedSmsSpam$train)
dim(partitionedSmsSpam$test)

partitionedSmsSpamTF <- prepareTfIdfWithLabels(partitionedSmsSpam, sparseLevel=0.95)
partitionedSmsSpamTF_98 <- prepareTfIdfWithLabels(partitionedSmsSpam, sparseLevel=0.98)
dim(partitionedSmsSpamTF_98$cleanedTrainMatrix)


dim(partitionedSmsSpamTF$cleanedTrainMatrix)
length(partitionedSmsSpamTF$cleanedTrainLabels)
dim(partitionedSmsSpamTF$cleanedTestMatrix)
length(partitionedSmsSpamTF$cleanedTestLabels)
tree_number

smsSpamTrainLabels <- partitionedSmsSpamTF$cleanedTrainLabels

smsSpamRfModel <- randomForest(x=as.matrix(partitionedSmsSpamTF$cleanedTrainMatrix), y=smsSpamTrainLabels, ntree=tree_number, keep.forest=TRUE)
smsSpamRfModel

predictSimple(smsSpamRfModel, partitionedSmsSpamTF$cleanedTestMatrix, partitionedSmsSpamTF$cleanedTestLabels)

smsSpamSVMModel <- svm(x=as.matrix(partitionedSmsSpamTF$cleanedTrainMatrix), y=smsSpamTrainLabels)
smsSpamSVMModel

predictSimple(smsSpamSVMModel, partitionedSmsSpamTF$cleanedTestMatrix, partitionedSmsSpamTF$cleanedTestLabels)

######################################### save train/test data in standard form #######################################
sms_train_for_python = as.matrix(partitionedSmsSpamTF$cleanedTrainMatrix)
sms_train_labels_for_python = as.integer(partitionedSmsSpamTF$cleanedTrainLabel)-1
sms_train_labels_for_python

sms_test_for_python =  as.matrix(partitionedSmsSpamTF$cleanedTestMatrix)
sms_test_labels_for_python = as.integer(partitionedSmsSpamTF$cleanedTestLabels)-1
sms_test_labels_for_python

write.csv(sms_train_for_python, file= '/Users/mjankowski/doc/data/smsSpam/for_python/smsSpam.data.train.csv', row.names = FALSE)
write(sms_train_labels_for_python, file='/Users/mjankowski/doc/data/smsSpam/for_python/smsSpam.labels.train.csv',
      ncolumns=1,sep="\n")

write.csv(sms_test_for_python, file= '/Users/mjankowski/doc/data/smsSpam/for_python/smsSpam.data.test.csv', row.names = FALSE)
write(sms_test_labels_for_python, file= '/Users/mjankowski/doc/data/smsSpam/for_python/smsSpam.labels.test.csv',
      ncolumns=1,sep="\n")


######################################### experiment up to 1000 topics ##############################################

smsSpamRange <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000)
smsSpamLdaModels <- lapply(smsSpamRange, function(x){calculateLDA(tfData = partitionedSmsSpamTF, topic_number = x)})

smsSpamTrainLabels <- partitionedSmsSpamTF$cleanedTrainLabels
smsSpamTestLabels <- partitionedSmsSpamTF$cleanedTestLabels

smsSpamRfModels <- lapply(smsSpamLdaModels, function(x){
  trainRfOnLda(x, smsSpamTrainLabels)
})

smsSpamRfErrors <- sapply(1:length(smsSpamRange), function(i){
  predictSimple(smsSpamRfModels[[i]], smsSpamLdaModels[[i]]$ldaTestData, smsSpamTestLabels)
})

smsSpamSVMModels <- lapply(smsSpamLdaModels, function(x){
  trainSVMOnLda(x, smsSpamTrainLabels)
})

smsSpamSVMErrors <- sapply(1:length(smsSpamRange), function(i){
  predictSimple(smsSpamSVMModels[[i]], smsSpamLdaModels[[i]]$ldaTestData, smsSpamTestLabels)
})

plot(smsSpamRange, smsSpamSVMErrors, type="l")
plot(smsSpamRange, smsSpamRfErrors, type="l")
min(smsSpamSVMErrors)
smsSpamRange[which.min(smsSpamSVMErrors)]

min(smsSpamRfErrors)
smsSpamRange[which.min(smsSpamRfErrors)]

######################################### additional models ##############################################

smsSpamAdditionalRange <- c(220, 240, 260, 280, 320, 340, 360, 380, 400, 420, 440, 460, 480, 550,650, 700, 750, 800, 850, 900, 950)

smsSpamAdditionalLdaModels <- lapply(smsSpamAdditionalRange, function(x){calculateLDA(tfData = partitionedSmsSpamTF, topic_number = x)})

smsSpamAdditionalRfModels <- lapply(smsSpamAdditionalLdaModels, function(x){
  trainRfOnLda(x, smsSpamTrainLabels)
})

smsSpamAdditionalRfErrors <- sapply(1:length(smsSpamAdditionalRange), function(i){
  predictSimple(smsSpamAdditionalRfModels[[i]], smsSpamAdditionalLdaModels[[i]]$ldaTestData, smsSpamTestLabels)
})

smsSpamAdditionalSVMModels <- lapply(smsSpamAdditionalLdaModels, function(x){
  trainSVMOnLda(x, smsSpamTrainLabels)
})

smsSpamAdditionalSVMErrors <- sapply(1:length(smsSpamAdditionalRange), function(i){
  predictSimple(smsSpamAdditionalSVMModels[[i]], smsSpamAdditionalLdaModels[[i]]$ldaTestData, smsSpamTestLabels)
})

plot(smsSpamAdditionalRange, smsSpamAdditionalSVMErrors, type="l")
plot(smsSpamAdditionalRange, smsSpamAdditionalRfErrors, type="l")
min(smsSpamAdditionalSVMErrors)
smsSpamAdditionalRange[which.min(smsSpamAdditionalSVMErrors)]

min(smsSpamAdditionalRfErrors)
smsSpamAdditionalRange[which.min(smsSpamAdditionalRfErrors)]


######################################## all models ###############################################


smsSpamLdaModelsExtended <- c(smsSpamLdaModels[1:23], smsSpamAdditionalLdaModels[1:4], smsSpamLdaModels[24], 
                                  smsSpamAdditionalLdaModels[5:8], smsSpamLdaModels[25], smsSpamAdditionalLdaModels[10:13],
                                  smsSpamLdaModels[26], smsSpamAdditionalLdaModels[14], smsSpamLdaModels[27], 
                                  smsSpamAdditionalLdaModels[15:20], smsSpamLdaModels[28])

smsSpamExtendedRangeList <- lapply(smsSpamLdaModelsExtended, function(m){
  m$topicmodel@k
})
smsSpamRangeExtended <- unlist(smsSpamExtendedRangeList)
smsSpamRangeExtended

smsSpamRfModelsExtended <- lapply(smsSpamLdaModelsExtended, function(x){
  trainRfOnLda(x, smsSpamTrainLabels)
})

smsSpamRfErrorsExtended <- sapply(1:length(smsSpamRangeExtended), function(i){
  predictSimple(smsSpamRfModelsExtended[[i]], smsSpamLdaModelsExtended[[i]]$ldaTestData, smsSpamTestLabels)
})

smsSpamSVMModelsExtended <- lapply(smsSpamLdaModelsExtended, function(x){
  trainSVMOnLda(x, smsSpamTrainLabels)
})

smsSpamSVMErrorsExtended <- sapply(1:length(smsSpamRangeExtended), function(i){
  predictSimple(smsSpamSVMModelsExtended[[i]], smsSpamLdaModelsExtended[[i]]$ldaTestData, smsSpamTestLabels)
})

plot(smsSpamRangeExtended, smsSpamSVMErrorsExtended, type="l")
plot(smsSpamRangeExtended, smsSpamRfErrorsExtended, type="l")

smsSpamSVMErrorsExtended
min(smsSpamSVMErrorsExtended)
max(smsSpamSVMErrorsExtended)
cbind(smsSpamRangeExtended, smsSpamRfErrorsExtended)
######################################### Estimate topic count max posterior ENTROPY ##############################################

smsSpamRange_300 <- smsSpamRange[1:24]
smsSpamAvgEntropyForPosterior_300 <- unlist(lapply(smsSpamLdaModels[1:24], avgEntropyForPosterior1))
smsSpamAvgEntropyForPosteriorRescaled_300 <- rescale(smsSpamAvgEntropyForPosterior_300)
smsSpamRfErrorsVectorRescaled_300 <- rescale(smsSpamRfErrors[1:24])
smsSpamSVMErrorsVectorRescaled_300 <- rescale(smsSpamSVMErrors[1:24])

length(smsSpamRange_300)
length(smsSpamAvgEntropyForPosteriorRescaled_300)
length(smsSpamRfErrorsVectorRescaled_300)
length(smsSpamSVMErrorsVectorRescaled_300)


smsSpamPosteriorEntropyWithClassificationErrors_300 <- data.frame(smsSpamRange_300,  
                                                                  smsSpamAvgEntropyForPosteriorRescaled_300, 
                                                                  smsSpamRfErrorsVectorRescaled_300,
                                                                  smsSpamSVMErrorsVectorRescaled_300)
dim(smsSpamPosteriorEntropyWithClassificationErrors_300)
smsSpamPosteriorEntropyWithClassificationErrorsMelted_300 <- melt(data = smsSpamPosteriorEntropyWithClassificationErrors_300, 
                                                                  id.vars = "smsSpamRange_300")

dev.new()
ggplot(data = smsSpamPosteriorEntropyWithClassificationErrorsMelted_300, aes(x = smsSpamRange_300, y = value, 
  color = factor(variable, labels = c("Average entropy of topics proportions", "Random Forest error", "SVM Error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)


######################################### MAX ENTROPY with posterior ##############################################

analysePosteriorEntropy(smsSpamRange_300, smsSpamLdaModels[1:24])

analysePosteriorEntropy(smsSpamRange, smsSpamLdaModels, header="Sms Spam")

######################################### impurity measure ###############################################################

library(arules)
library(infotheo)
noBuckets <- 10

conditionalEntropy(smsSpamLdaModels[[12]], smsSpamTrainLabels, bins=10)

joinedEntropy(smsSpamLdaModels[[20]], smsSpamTrainLabels, bins=10)



condEntropy(smsSpamLdaModels[[12]])

tmptrainData <- smsSpamLdaModels[[12]]$ldaTrainData
dim(smsSpamLdaModels[[12]]$ldaTrainData)
tmpnRows <- dim(tmptrainData)[1]
tmpnCols <- dim(tmptrainData)[2]
tmpRawDisc <- arules::discretize(tmptrainData, method = "frequency", categories = 10, onlycuts=FALSE)
tmpTrainDisc <- matrix(as.numeric(tmpRawDisc), nrow = tmpnRows, ncol=tmpnCols)
H <- infotheo::condentropy(X=smsSpamTrainLabels, Y=tmpTrainDisc, method = "emp")
H
length(trainLabels)

ttt <- as.numeric(table(smsSpamTrainLabels))
ttt
entropy.plugin(ttt/sum(ttt))


data(USArrests)

ddd <- data.frame(tmpTrainDisc, smsSpamTrainLabels)
dim(ddd)
dim(tmpTrainDisc)

H <- infotheo::entropy(infotheo::discretize(ddd),method="emp")
H

######################################### Average entropy ##############################################
subrange = 1:28
analyseAverageEntropy(smsSpamRange[subrange], smsSpamLdaModels[subrange], smsSpamRfErrors[subrange], 
                      smsSpamSVMErrors[subrange], rescale=TRUE, header="Sms Spam")

######################################### Estimate topic count conditional entropy ENTROPY ##############################################
length(smsSpamRange)

mutualInfo(smsSpamLdaModels[[3]], smsSpamTrainLabels, bins=10)

subrange = 3:28
analyseJoinedEntropy(smsSpamRange[subrange], smsSpamLdaModels[subrange], smsSpamTrainLabels, smsSpamRfErrors[subrange], 
                     smsSpamSVMErrors[subrange], header="Sms Spam, bins=100", bins=100)

smsMutualInfoForPosterior <- unlist(lapply(smsSpamLdaModels[subrange], function(model) {
  mutualInfo(model, smsSpamTrainLabels, bins=10)
}))

smsMutualInfoForPosterior_2 <- avg_mi_all_models(smsSpamLdaModels[subrange], smsSpamTrainLabels)

dev.new()
plot(smsSpamRange[subrange], smsMutualInfoForPosterior_2, type="l")

conditionalEntropy2(smsSpamLdaModels[[10]], smsSpamTrainLabels, bins=10)

subrange = 3:28
analyseConditionalEntropy(smsSpamRange[subrange], smsSpamLdaModels[subrange], smsSpamTrainLabels, smsSpamRfErrors[subrange], 
                          smsSpamSVMErrors[subrange], header="Sms Spam, bins=2", bins=2, rescale=FALSE, runOnTest = FALSE)

infotheo::entropy(smsSpamTrainLabels)


discretizeAndEntropy <- function(ldaModel, bins){
  trainData <- ldaModel$ldaTrainData
  nRows <- dim(trainData)[1]
  nCols <- dim(trainData)[2]
  rawDisc <- arules::discretize(trainData, method = "frequency", categories = bins, onlycuts=FALSE)
  trainDisc <- matrix(as.numeric(rawDisc), nrow = nRows, ncol=nCols)
 
  trainDiscEntropy <- infotheo::entropy(trainDisc, method = "emp")
  trainDiscEntropy
}

entropyAllModels <- function(models, bins){
  discEntropies <- lapply(smsSpamLdaModels, function(m){
    discretizeAndEntropy(m, bins=bins)
  })
  discEntropies
}
entropyAllModelsAllBins <- function(models, bins=c(2,5,10,50,100)){
  lapply(bins, function(b){
    entropyAllModels(models, b)
  })
}

charts <- entropyAllModelsAllBins(smsSpamLdaModels)
length(charts)

aaaaa <- data.frame(smsSpamRange, unlist(charts[[1]]),
                    unlist(charts[[2]]),
                           unlist(charts[[3]]),
                                  unlist(charts[[4]]),
                                         unlist(charts[[5]]))

unlist(charts[[5]])

aaaMelted <- melt(data = aaaaa, 
                  id.vars = "smsSpamRange")

dev.new()
ggplot(data = aaaMelted, aes(x = smsSpamRange, y = value, 
  color = variable)) + 
  geom_point() + geom_line(size=1) + 
 theme_classic(base_size = 18)

testAAA <- function(dim){
  d <- sapply(1:dim, function(x){rnorm(742)})
  disc <- arules::discretize(d, method = "interval", categories = 10000, onlycuts=FALSE)
  infotheo::entropy(disc, method = "emp")
}

testAAA(10)

sapply(1:200, function(x){log(x)})

######################################### Pareto front for topics correlation and topics entropy ##############################################

######################################### Analyse ensemble #########################################

chooseModelUsingPerplexity(partitionedSmsSpamTF$cleanedTestMatrix, smsSpamLdaModels)  
#chosenModels <- chooseModelUsingPerplexity(partitionedSmsSpamTF$cleanedTestMatrix, smsSpamLdaModels)

smsSpamLdaModelsExtended
chosenModels <- chooseModelUsingPerplexity(partitionedSmsSpamTF$cleanedTestMatrix, smsSpamLdaModelsExtended)

dim(partitionedSmsSpamTF$cleanedTrainMatrix)
smsSpamFirstDocument <- partitionedSmsSpamTF$cleanedTestMatrix[2,]
smsSpamFirstDocument

smsSpamLdaModelsExtended[1]$topicmodel

logLiks

perplexities_for_smsSpamFirstDocument <- lapply(smsSpamLdaModelsExtended, function(m){
  p <- perplexity(m$topicmodel, smsSpamFirstDocument)
  p
})



likelihoods_for_smsSpam <- lapply(smsSpamLdaModelsExtended, function(m){
  m$topicmodel@loglikelihood
})

likelihoods_for_smsSpam_df = data.frame(matrix(unlist(likelihoods_for_smsSpam), nrow=742, byrow=F))
dim(likelihoods_for_smsSpam_df)

likelihoods_for_smsSpam[[1]]
likelihoods_for_smsSpam_df[,1]

apply(likelihoods_for_smsSpam_df, 1, FUN=which.max)

  
smsSpamFirstDocument

perplexities_for_smsSpamFirstDocument

plot(smsSpamRangeExtended[0:30], perplexities_for_smsSpamFirstDocument[0:30])

######################################### Analyse mutual information #########################################

smsSpamSubRange=2:28
dev.new()
analyseMutualInformation(smsSpamRange[smsSpamSubRange], smsSpamLdaModels[smsSpamSubRange], smsSpamTrainLabels, 
                         smsSpamRfErrors[smsSpamSubRange], smsSpamSVMErrors[smsSpamSubRange], 
                         header="Sms Spam, bins=2", bins=2, rescale=TRUE)

######################################### Analyse mutual information on test #########################################

smsSpamSubRange=2:28
dev.new()
analyseMutualInformation(smsSpamRange[smsSpamSubRange], smsSpamLdaModels[smsSpamSubRange], smsSpamTrainLabels, 
                         smsSpamRfErrors[smsSpamSubRange], smsSpamSVMErrors[smsSpamSubRange], 
                         header="Sms Spam, bins=2", bins=2, rescale=TRUE)

######################################### Analyse mutual information #########################################

smsSpamSubRangeExtended=1:47
analyseMutualInformation(smsSpamRangeExtended[smsSpamSubRangeExtended], smsSpamLdaModelsExtended[smsSpamSubRangeExtended], smsSpamTrainLabels, 
                         smsSpamRfErrorsExtended[smsSpamSubRangeExtended], smsSpamSVMErrorsExtended[smsSpamSubRangeExtended], 
                         header="Sms Spam, bins=2", bins=2, rescale=TRUE)

cbind(1:28, smsSpamRange)

betaDistribution <- exp(smsSpamLdaModels[[13]]$topicmodel@beta)
dim(betaDistribution)
delta <- diri.est(betaDistribution)

smsSpamLdaModels[[13]]$topicmodel@k


######################################### Compare models using per-class KL between original and sampled data #########################################

smsSubrangeForSampling = 1:28
smsTrainMatrix <- as.matrix(partitionedSmsSpamTF$cleanedTrainMatrix)

smsDocLengths <- rowSums(smsTrainMatrix)
smsSamplesFromLdaModels <- lapply(smsSpamLdaModels[smsSubrangeForSampling], function(m){
  generateCorpusFromModel(m, smsDocLengths)
})



smsKlAvgs <- calculateAvgKl(range = smsSpamRange[smsSubrangeForSampling], samplesFromModel = smsSamplesFromLdaModels,
                                labels = smsSpamTrainLabels, tfMatrix = partitionedSmsSpamTF$cleanedTrainMatrix)
dev.new()
plot(smsSpamRange[smsSubrangeForSampling], smsKlAvgs, type="l")

chi = calculatePValueBetweenDataAndSampleForClass(
  partitionedSmsSpamTF$cleanedTrainMatrix, 
  smsSpamLdaModels[smsSubrangeForSampling],
  smsSamplesFromLdaModels[1:2],
  labels = smsSpamTrainLabels
)
sapply(chi, function(c){c$p.value})
sapply(chi, function(c){c$statistic})

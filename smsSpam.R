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

smsSpamMaxEntropy <- sapply(smsSpamRange_300, log)
smsSpamPosteriorEntropy <- unlist(smsSpamAvgEntropyForPosterior_300)
smsSpamMaxEntropyWithPosterior <- data.frame(smsSpamRange_300,  
                                      smsSpamPosteriorEntropy, 
                                      smsSpamMaxEntropy)
smsSpamMaxEntropyWithPosteriorMelted <- melt(data = smsSpamMaxEntropyWithPosterior, id.vars = "smsSpamRange_300")
dev.new()
ggplot(data = smsSpamMaxEntropyWithPosteriorMelted, aes(x = smsSpamRange_300, y = value, 
  color = factor(variable, labels = c("posterior entropy", "max entropy")), linetype=variable)) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)




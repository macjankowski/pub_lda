
library(entropy)    
library (lda)
library(MASS)
library(scales)
library(ggplot2)
library(randomForest)

source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')
source('./ensemble.R')


tree_number <- 500
topic_number <- 10

filePath = '/Users/mjankowski/doc/data/apps/apps_desc.csv'

data <- read.csv(filePath, sep=";")  
dim(data)

dataAll <- cleanData(data)

labelMapping <- data.frame(label = c("ANDROID_TOOL", "KEYBOARD", "GAME", "NONE", "WIDGET", "USE_INTERNET", 
                                       "DOCUMENT_EDITOR", "LOCATE_POSITION", "APP_LIBRARY", "INTERNET_BROWSER", 
                                       "MESSAGING", "WALLPAPER", "WEATHER", "USE_CONTACTS", "BACKUP", "WORKOUT_TRACKING", 
                                       "CALENDAR", "MONEY", "GPS_NAVIGATION", "FLASHLIGHT", "HOME_LOCK_SCREEN", 
                                       "SMS", "JOB_SEARCH", "EBANKING", "CONTACT_MANAGER"), label = c(0:24))

dataAllLabelsNumeric <- labelsToNumeric(dataAll, labelMapping)
names(dataAllLabelsNumeric)[names(dataAllLabelsNumeric) == "label.1"] = "label"

dim(dataAllLabelsNumeric)
partitioned <- partitionData(dataAllLabelsNumeric)
dim(partitioned$train)
dim(partitioned$test)

tfidfData <- prepareTfIdfWithLabels(partitioned, sparseLevel=0.98)

dim(tfidfData$cleanedTrainMatrix)
length(tfidfData$cleanedTrainLabels)
dim(tfidfData$cleanedTestMatrix)
length(tfidfData$cleanedTestLabels)
tree_number

res_full <- trainAndPredict(tree_number, tfidfData$cleanedTrainMatrix, tfidfData$cleanedTrainLabels, 
                            tfidfData$cleanedTestMatrix, tfidfData$cleanedTestLabels)

plotResults(res_full$testResult$threshold, res_full$testResult$bridgeRatio, res_full$testResult$errorRatio)

res_full$model

######################################### LSA 10 topics ##############################################

lsa_matrices <- calculateLSA(tfidfData, topic_number=32)
lsa.training.set.tfidf <- lsa_matrices$lsaTrainData
lsa.testing.set.tfidf <- lsa_matrices$lsaTestData

res_lsa <- trainAndPredict(tree_number=2000, 
                           trainData=lsa.training.set.tfidf, trainLabels=tfidfData$cleanedTrainLabels, 
                           testData=lsa.testing.set.tfidf, testLabels=tfidfData$cleanedTestLabels)

dev.new()
plotResults(res_lsa$testResult$threshold, res_lsa$testResult$bridgeRatio, res_lsa$testResult$errorRatio)

res_lsa$model

######################################### PLSA 10 topics ##############################################

plsa_10 <- fast_plsa(as.matrix(tfidfData$cleanedTrainMatrix), topic_number)

######################################### LDA 10 topics ##############################################

lda_10 <- calculateLDA(tfidfData, 10)
posterior(lda_10$topicmodel)[2]

res_lda_10 <- trainAndPredict(tree_number, lda_10$ldaTrainData, tfidfData$cleanedTrainLabels, 
                              lda_10$ldaTestData, tfidfData$cleanedTestLabels)


plotResults(res_lda_10$testResult$threshold, res_lda_10$testResult$bridgeRatio, res_lda_10$testResult$errorRatio)

res_lda_10$model


######################################### LDA 30 topics ##############################################


lda_30 <- calculateLDA(tfidfData, 30)
posterior(lda_30$topicmodel)[2]

tree_number <- 100
res_lda_30 <- trainAndPredict(tree_number, lda_30$ldaTrainData, tfidfData$cleanedTrainLabels, 
                           lda_30$ldaTestData, tfidfData$cleanedTestLabels)

res_lda_30$model

out <- predict(res_lda_30$model, as.matrix(lda_30$ldaTestData), type="prob")
confidence_value <- apply(out,1, max)
confidence_value

length(confidence_value)

dev.new()
hist(confidence_value*100, breaks=100)

#out <- distributionOfPrediction(res_lda_30$model, as.matrix(lda_30$ldaTestData))



plotResults(res_lda_30$testResult$threshold, res_lda_30$testResult$bridgeRatio, res_lda_30$testResult$errorRatio)

res_lda_30$model

######################################### Estimate topic count using LSA ##############################################

res <- estimateTopicsCountLSA(2,1000,5, tfidfData = tfidfData, tree_number = tree_number)

topics = seq(from = 2, to = 1000, by = 5)

errors <- res$rfResult
dev.new()
plot(topics, errors, type = "l", main = "Classification performance for dirrerent values of topic number", xlab = "Topics number", 
     ylab = "Error rate on test data", col="black", col.axis = "black", col.lab = "black")

errors_vect <- unlist(errors)
lsa_errors <- errors_vect

max_index <- which(errors_vect == max(errors_vect[8:length(errors_vect)]))
topics_number_for_max <- topics[max_index]
max_value <- errors_vect[max_index]
max_value

min_index <- which(errors_vect == min(errors_vect))
min_index
min_value <- errors_vect[min_index]
min_value

######################################### Estimate topic count all ##############################################

range = c(2,5,10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 1000)

all_res <- estimateTopicsCount4MethodsRange(range, tfidfData = tfidfData)

all_topics = range
all_result <- all_res$ldatuningResults

dev.new()
plot(all_result$topics, all_result$Griffiths2004,type = "l", col="black", xlab = "Topics number", ylab="logP(C|K)", main = "")

dev.new()
plot(all_result$topics, all_result$CaoJuan2009,type = "l", col="black", xlab = "Topics number", 
     ylab="Average cosine similarity", main = "", )

dev.new()
plot(all_result$topics, all_result$Arun2010,type = "l", col="black", xlab = "Topics number", 
     ylab="A(theta,beta)", main = "")

dev.new()
plot(all_result$topics, all_result$Deveaud2014,type = "l", col="black", xlab = "Topics number", 
     ylab="Avg. Jensen-Shannon Divergence", main = "")

######################################### Estimate topic count 4 methods ##############################################

res <- estimateTopicsCount4Methods(2,100,5, tfidfData = tfidfData)

topics = seq(from = 2, to = 100, by = 5)

errors <- res$ldatuningResults

result <- res$ldatuningResults
dev.new()
plot(result$topics, result$Griffiths2004,type = "l", col="black", xlab = "", ylab="",main = "Classification performance for dirrerent values of topic number")
par(new=TRUE)
plot(result$topics, result$CaoJuan2009,type = "l", col="red", xlab = "", ylab="")
par(new=TRUE)
plot(result$topics, result$Arun2010,type = "l", col="green", xlab = "", ylab="")
par(new=TRUE)
plot(result$topics, result$Deveaud2014,type = "l", col="violet", xlab = "", ylab="")



dev.new()
plot(topics, errors, type = "l", main = "Classification performance for dirrerent values of topic number", xlab = "Topics number", 
     ylab = "Error rate on test data", col="black", col.axis = "black", col.lab = "black")


######################################### Train models ##############################################

range = c(2,5,10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 1000)
minimal_range = c(2,10)
small_range = c(2,5,10, 15, 20, 30, 50, 75, 100)

models <- lapply(range, function(x){calculateLDA(tfData = tfidfData, topic_number = x)})
ldaModels <- models

####################################### train additional models #############################################

additional_range = c(40, 60, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190)

additional_models <- lapply(additional_range, function(x){calculateLDA(tfData = tfidfData, topic_number = x)})


######################################### Estimate topic count ENTROPY ##############################################

avgEntropyForModels <- lapply(models, avgEntropyForModel)
avgEntropyForModels <- unlist(avgEntropyForModels)
avgEntropyForModels

plot(range,avgEntropyForModels, type = "l")

######################################### Scores for Random Forests ##############################################


trainLabels <- tfidfData$cleanedTrainLabels
testLabels <- tfidfData$cleanedTestLabels

trainAndPredictRf <- function(model){
  trainData <- model$ldaTrainData
  testData <- model$ldaTestData
  trainAndPredictSimple(tree_number, trainData, trainLabels, testData, testLabels)
}

accuraciesList <- lapply(models, trainAndPredictRf)
accuracies <- as.numeric(accuraciesList)
lines(range,accuracies*10, type = "l")

######################################### Scores for Random Forests All models ##############################################
additional_range = c(40, 60, 80, 90, 110, 120, 130, 140, 160, 170, 180, 190)
range = c(2,5,10, 15, 20, 30, 50, 75, 100, 150, 200, 300, 400, 500, 600, 1000)
all_range <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000)

range_200 <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
lda_models_200 <- list(ldaModels[[1]], ldaModels[[2]], ldaModels[[3]], ldaModels[[4]], ldaModels[[5]], ldaModels[[6]],
                       additional_models[[1]], ldaModels[[7]], additional_models[[2]],ldaModels[[8]], 
                       additional_models[[3]], additional_models[[4]], ldaModels[[9]], additional_models[[5]], 
                       additional_models[[6]], additional_models[[7]], additional_models[[8]], ldaModels[[10]], 
                       additional_models[[9]], additional_models[[10]], additional_models[[11]], additional_models[[12]],
                       ldaModels[[11]])

all_models <- list(ldaModels[[1]], ldaModels[[2]], ldaModels[[3]], ldaModels[[4]], ldaModels[[5]], ldaModels[[6]],
                   additional_models[[1]], ldaModels[[7]], additional_models[[2]],ldaModels[[8]], 
                   additional_models[[3]], additional_models[[4]], ldaModels[[9]], additional_models[[5]], 
                   additional_models[[6]], additional_models[[7]], additional_models[[8]], ldaModels[[10]], 
                   additional_models[[9]], additional_models[[10]], additional_models[[11]], additional_models[[12]],
                   ldaModels[[11]], ldaModels[[12]], ldaModels[[13]], ldaModels[[14]], ldaModels[[15]], ldaModels[[16]])
length(all_models)

trainLabels <- tfidfData$cleanedTrainLabels
testLabels <- tfidfData$cleanedTestLabels

trainAndPredictRf <- function(model){
  trainData <- model$ldaTrainData
  testData <- model$ldaTestData
  trainAndPredictSimple(tree_number, trainData, trainLabels, testData, testLabels)
}

accuraciesList <- lapply(all_models, trainAndPredictRf)

all_res <- estimateTopicsCount4MethodsRange(range, tfidfData = tfidfData)

accuraciesList_200 <- accuraciesList[1:23]
accuracies_200 <- as.numeric(accuraciesList_200)
dev.new()
plot(range_200,accuracies_200, type = "l")

accuracies <- as.numeric(accuraciesList)
dev.new()
plot(all_range,accuracies, type = "l")

####################################### train rf models #############################################




rfModels <- lapply(models, trainRfOnLda)
rfModel <- rfModels[[3]]
testData <- ldaModels[[3]]$ldaTestData

####################################### ensemble without score #############################################

res <- predictEnsemble(testLabels, rfModels, ldaModels)
errorForEnsembleResult(res, testLabels)

####################################### ensemble without score on best models only #############################################

res <- predictEnsemble(testLabels, rfModels, ldaModels)
errorForEnsembleResult(res, testLabels)

####################################### ensemble with score #############################################
res <- predictWithScoreEnsemble(testLabels, rfModels, ldaModels)
errorForEnsembleWithScoreResult(res$classes, res$scores, testLabels)

####################################### ensemble with score on best models only #############################################

subRange <- c(3,4,5,6,7,8,9)
accuracies[subRange]
subRfModels <- rfModels[subRange]
subLdaModels <- ldaModels[subRange]

res <- predictWithScoreEnsemble(testLabels, subRfModels, subLdaModels)
errorForEnsembleWithScoreResult(res$classes, res$scores, testLabels)

####################################### plot topic number scores with accuracies #############################################

dev.new()
plot(all_result$topics, all_result$Griffiths2004,type = "l", col="black", xlab = "Topics number", ylab="logP(C|K)", main = "")
par(new=TRUE)
plot(all_result$topics, all_result$CaoJuan2009,type = "l", col="black", xlab = "Topics number", 
     ylab="Average cosine similarity", main = "")
par(new=TRUE)
plot(all_result$topics, all_result$Arun2010,type = "l", col="black", xlab = "Topics number", 
     ylab="A(theta,beta)", main = "")
par(new=TRUE)
plot(all_result$topics, all_result$Deveaud2014,type = "l", col="black", xlab = "Topics number", 
     ylab="Avg. Jensen-Shannon Divergence", main = "")
par(new=TRUE)
plot(range, accuracies, type = "l", col="red")
par(new=TRUE)
plot(range,avgEntropyForModels, type = "l", col="blue")

####################################### ggplot topic number scores with accuracies #############################################

gryffith <- rescale(all_result$Griffiths2004)
cao <- rescale(all_result$CaoJuan2009)
arun <- rescale(all_result$Arun2010)
deveaud <- rescale(all_result$Deveaud2014)
random_forest_accuracy <- rescale(accuracies)

topics_number <- all_result$topics
df <- data.frame(topics_number, gryffith, cao, arun, deveaud, random_forest_accuracy)
df2 <- melt(data = df, id.vars = "topics_number")

dev.new()
ggplot(data = df2, aes(x = topics_number, y = value, linetype=variable)) + geom_line()


#################################### Linear discriminant analysis ############################################

 

library("Matrix") 
dtmTrain <- tfidfData$cleanedTrainMatrix
tfDataAsMatrix <- sparseMatrix(i=dtmTrain$i, j=dtmTrain$j, x=dtmTrain$v, dims=c(dtmTrain$nrow, dtmTrain$ncol))
as.matrix(tfDataAsMatrix)
lda.train.data <- data.frame(label=trainLabels, as.matrix(tfDataAsMatrix))

dtmTest <- tfidfData$cleanedTestMatrix
tfTestDataAsMatrix <- sparseMatrix(i=dtmTest$i, j=dtmTest$j, x=dtmTest$v, dims=c(dtmTest$nrow, dtmTest$ncol))



lda.model <- lda(label ~ .,data = lda.train.data)

lda.test.data <- data.frame(as.matrix(tfTestDataAsMatrix))

lda.prediction <- predict(lda.model, lda.test.data)
lda.prediction.class <- lda.prediction$class

attributes(testLabels)$levels
lda.prediction.class$levels = attributes(testLabels)$levels
lda.prediction.correct <- lda.prediction.class[lda.prediction.class == testLabels]
error <- 1-length(lda.prediction.correct)/length(testLabels)
error

length(lda.prediction.class)

lda.prediction.class

######################################### experiment up to 200 topics ##############################################

range_200 <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
lda_models_200 <- list(ldaModels[[1]], ldaModels[[2]], ldaModels[[3]], ldaModels[[4]], ldaModels[[5]], ldaModels[[6]],
                       additional_models[[1]], ldaModels[[7]], additional_models[[2]],ldaModels[[8]], 
                       additional_models[[3]], additional_models[[4]], ldaModels[[9]], additional_models[[5]], 
                       additional_models[[6]], additional_models[[7]], additional_models[[8]], ldaModels[[10]], 
                       additional_models[[9]], additional_models[[10]], additional_models[[11]], additional_models[[12]],
                       ldaModels[[11]])

trainLabels <- tfidfData$cleanedTrainLabels
testLabels <- tfidfData$cleanedTestLabels

trainAndPredictRf <- function(model){
  trainData <- model$ldaTrainData
  testData <- model$ldaTestData
  trainAndPredictSimple(tree_number, trainData, trainLabels, testData, testLabels)
}



topic_number_200 <- estimateTopicsCount4MethodsRange(range_200, tfidfData = tfidfData)

accuraciesList_200 <- lapply(lda_models_200, trainAndPredictRf)
accuracies_200 <- as.numeric(accuraciesList_200)
dev.new()
plot(range_200,accuracies_200, type = "l")

####################################### ggplot 2-200 topic number scores with accuracies #############################################

#avgEntropyForModels_200 <- lapply(lda_models_200, avgEntropyForModel)
#avgEntropyForModels_200 <- rescale(unlist(avgEntropyForModels_200))

tuning_200 <- topic_number_200$ldatuningResults

gryffith_200 <- rescale(tuning_200$Griffiths2004)
cao_200 <- rescale(tuning_200$CaoJuan2009)
arun_200 <- rescale(tuning_200$Arun2010)
deveaud_200 <- rescale(tuning_200$Deveaud2014)
random_forest_accuracy_200 <- rescale(accuracies_200)
#perplexities_200 <- rescale(sapply(lda_models_200, function(x){ perplexity(x$topicmodel, tfidfData$cleanedTestMatrix)}))

topics_number_200 <- tuning_200$topics
#df_200 <- data.frame(topics_number_200, gryffith_200, cao_200, arun_200, deveaud_200, random_forest_accuracy_200, avgEntropyForModels_200)
df_200 <- data.frame(topics_number_200, gryffith_200, cao_200, arun_200, deveaud_200, random_forest_accuracy_200)
df2_200 <- melt(data = df_200, id.vars = "topics_number_200")

dev.new()
ggplot(data = df2_200, aes(x = topics_number_200, y = value, linetype=variable)) + geom_line()

####################################### mutual information #############################################
library(arules)

avg_mi_all <- avg_mi_all_models(lda_models_200, trainLabels)
plot(range_200, avg_mi_all, type="l")

all_models_avg_spearman <- avg_spearman_all_models(lda_models_200, trainLabels)
plot(range_200, all_models_avg_spearman, type="l")

all_models_avg_spearman

avg_spearman_single_model_without_mean(lda_models_200[[2]]$ldaTrainData, (1:length(trainLabels)))

max(abs(harmonic.mean(avg_spearman_single_model_without_mean(lda_models_200[[2]]$ldaTrainData, trainLabels))))




####################################### choose model based on likelihood #############################################

posterior <- posterior(ldaModels[[6]]$topicmodel, tfidfData$cleanedTestMatrix)
posterior[[2]] # is the matrix with topics as cols, new documents as rows and cell values as posterior probabilities
posterior$topics
  
perplexities200_not_scaled <- sapply(lda_models_200, function(x){ perplexity(x$topicmodel, tfidfData$cleanedTestMatrix)})

perplexities200_not_scaled_train <- sapply(lda_models_200, function(x){ perplexity(x$topicmodel, tfidfData$cleanedTrainMatrix)})

plot(range_200, perplexities200_not_scaled_train, type="l")

gryffith_200 <- rescale(tuning_200$Griffiths2004)
perp_200_train_scaled <- rescale(perplexities200_not_scaled)

topics_number_200 <- tuning_200$topics
df_200_train_gryffith_parp <- data.frame(topics_number_200, gryffith_200, perp_200_train_scaled)
df_200_train_gryffith_parp <- melt(data = df_200_train_gryffith_parp, id.vars = "topics_number_200")

dev.new()
ggplot(data = df_200_train_gryffith_parp, aes(x = topics_number_200, y = value, linetype=variable)) + geom_line()



####################################### ensemble with score on best models only #############################################

subRange <- c(3,4,5,6,7,8,9)
accuracies[subRange]
subRfModels <- rfModels[subRange]
subLdaModels <- ldaModels[subRange]

res <- predictWithScoreEnsemble(subRfModels, subLdaModels)
errorForEnsembleWithScoreResult(res$classes, res$scores, testLabels)

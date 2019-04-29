
library(reshape2)
library(ggplot2)
library(ggplot2)
library(scales)


######################################### experiment up to 200 topics ##############################################

range_200 <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200);
lda_models_200 <- lapply(range_200, function(x){calculateLDA(tfData = tfidfData, topic_number = x)})

trainRfOnLda <- function(ldaModel, trainLabels, tree_number=2000){
  trainData <- ldaModel$ldaTrainData
  testData <- ldaModel$ldaTestData
  rfModel <- randomForest(x=as.matrix(trainData), y=trainLabels, ntree=tree_number, keep.forest=TRUE)
  rfModel
}

rfModels_200 <- lapply(lda_models_200, function(x){
  trainRfOnLda(x, trainLabels)
})

testLabels <- tfidfData$cleanedTestLabels
length(testLabels)
length(trainLabels)

accuracies_200 <- sapply(1:23, function(i){
  predictSimple(rfModels_200[[i]], lda_models_200[[i]]$ldaTestData, testLabels)
})
accuracies_200
cbind(range_200, accuracies_200)[subRange,]
dev.new()
recognizedClassesForHistogram(testLabels, rfModels_200, lda_models_200, topicNumbers = range_200)

####################################### ggplot 2-200 topic number scores with accuracies #############################################

topic_number_200 <- estimateTopicsCount4MethodsRange(range_200, tfidfData = tfidfData)

tuning_200 <- topic_number_200$ldatuningResults

gryffith_200 <- rescale(tuning_200$Griffiths2004)
cao_200 <- rescale(tuning_200$CaoJuan2009)
arun_200 <- rescale(tuning_200$Arun2010)
deveaud_200 <- rescale(tuning_200$Deveaud2014)
random_forest_accuracy_200 <- rescale(accuracies_200)
random_forest_error_200 <-random_forest_accuracy_200
#perplexities_200 <- rescale(sapply(lda_models_200, function(x){ perplexity(x$topicmodel, tfidfData$cleanedTestMatrix)}))

topics_number_200 <- tuning_200$topics
#df_200 <- data.frame(topics_number_200, gryffith_200, cao_200, arun_200, deveaud_200, random_forest_accuracy_200, avgEntropyForModels_200)
df_200 <- data.frame(topics_number_200, gryffith_200, cao_200, arun_200, deveaud_200, random_forest_error_200)
df2_200 <- melt(data = df_200, id.vars = "topics_number_200")

dev.new()
#ggplot(data = df2_200, aes(x = topics_number_200, y = value, linetype=variable, col=variable)) + geom_line() + 
#  labs(x = "Topics number", y="Value")
ggplot(data = df2_200, aes(x = topics_number_200, y = value, color = factor(variable, 
  labels = c("Likelihood",  "Cosine similarity", "Arun", "Deveaud", "Random Forest error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topics number", y="Value", color="Methods") + theme_classic(base_size = 18)

####################################### ensemble with score on best models only, substracted mean #############################################

subRange <- c(4, 5, 6, 8, 9, 10,11,12,13)
accuracies_200[subRange]
subRfModels_200 <- rfModels_200[subRange]
subLdaModels_200 <- lda_models_200[subRange]

res <- predictWithScoreEnsemble(subRfModels_200, subLdaModels_200)
errorForEnsembleWithScoreResult(res$classes, res$scores, testLabels)

######################################### models mean confidence ##############################################


p <- predictWithScoreEnsembleSubstractMean(testLabels, rfModels_200, lda_models_200)

meanScores <- sapply(1:length(rfModels_200), function(x){mean(p$scores[x,])})
dev.new()
plot(range_200, meanScores, type="l", xlab = "Topics number", ylab = "Average confidence")

p$scores
mean(p$scores[6,])

####################################### mutual information #############################################
library(arules)
library(dae)

avg_mi_all_with_const <- avg_mi_all_models(lda_models_200, rep(1, length(trainLabels)))
plot(range_200, avg_mi_all_with_const, type="l")
avg_mi_all <- avg_mi_all_models(lda_models_200, trainLabels)
plot(range_200, avg_mi_all, type="l")

avg_mi_all_with_const
avg_mi_all
avg_mi_all - avg_mi_all_with_const

######################################### Estimate topic count ENTROPY ##############################################
length(models)
more_models <- tail(models, n=5)
extendedModels <- c(lda_models_200, more_models)

avgEntropyForPosterior <- lapply(extendedModels, avgEntropyForPosterior1)
avgEntropyForPosteriorRescaled <- rescale(unlist(avgEntropyForPosterior))

avgEntropyForModels <- lapply(extendedModels, avgEntropyForModel)
avgEntropyForModels <- rescale(unlist(avgEntropyForModels))
avgEntropyForModels
average_entropy_of_topics <- avgEntropyForModels

moreAccuraciesList <- lapply(more_models, trainAndPredictRf)

rfModels_300_1000 <- lapply(more_models, function(x){
  trainRfOnLda(x, trainLabels)
})

moreAccuracies <- unlist(moreAccuraciesList)

random_forest_accuracy_all <- rescale(c(accuracies_200,moreAccuracies))
topics_number_with_more <- c(topics_number_200, c(300, 400, 500, 600, 1000))
topics_number <- topics_number_with_more

reviewsAvgEntropyForModelsRescaled <- rescale(unlist(reviewsAvgEntropyForModels))

df_all <- data.frame(topics_number, average_entropy_of_topics, avgEntropyForPosteriorRescaled, random_forest_accuracy_all)
df2_all <- melt(data = df_all, id.vars = "topics_number")

dev.new()
ggplot(data = df2_all, aes(x = topics_number, y = value, 
  color = factor(variable, labels = c("Average entropy of topics",  "Ang entropy posterior", "Random Forest error")))) + 
    geom_point() + geom_line(size=1) + 
    labs(x = "Topics number", y="Value", color="Variable") + theme_classic(base_size = 18)




####################################### lsa with rf error #############################################

lsa_topics_number = seq(from = 2, to = 1000, by = 5)
lsa_errors
length(topics_number_200)

random_forest_accuracy_all_not_scaled <- c(accuracies_200,moreAccuracies)

lsa_topics_number_and_rf <- sort(c(lsa_topics_number, topics_number_200))

random_forest_error_all_list <- approx(topics_number_with_more, random_forest_accuracy_all_not_scaled, xout=lsa_topics_number)$y

plot(lsa_topics_number, random_forest_error_all_list, type="l")

random_forest_error_all <- rescale(unlist(random_forest_error_all_list))
lda_error <- random_forest_error_all
lsa_error <- rescale(lsa_errors)

lsa_df_all <- data.frame(lsa_topics_number, lda_error, lsa_error)
lsa_df2_all <- melt(data = lsa_df_all, id.vars = "lsa_topics_number")

dev.new()
ggplot(data = lsa_df2_all, aes(x = lsa_topics_number, y = value, linetype=variable)) + geom_line() + 
  labs(x = "Topics number", y="Random Forest Error on LDA and LSA")

##################################### perplexity based ensemble ######################################

chosenModels <- chooseModelUsingPerplexity(tfidfData$cleanedTestMatrix, extendedModels)
chosenModels
length(chosenModels)
max(chosenModels)
min(chosenModels)

s <- sapply(1:23, function(x){
  l <- length(which(chosenModels == x))
  l
})
sum(unlist(s))

length(lda_models_200)

idxForIthModel <- which(chosenModels == 2)
ldaTestData <- lda_models_200[[i]]$ldaTestData[idxForIthModel,] #get data for ith model
predictSimple(rfModels_200[[i]], ldaTestData, testLabels[idxForIthModel])

length(subLdaModels_200)
chosenModels_best_9 <- chooseModelUsingPerplexity(tfidfData$cleanedTestMatrix, subLdaModels_200)
chosenModels_best_9
v <- errorPerplexityEnsemble(chosenModels, subLdaModels_200, subRfModels_200)
v

idxxxx <- which(chosenModels == 2)
length(idxxxx)

dim(lda_models_200[[i]]$ldaTestData[idxxxx,])

############################################ Perplexity on test set ##########################################


perModelPerplexity <- lapply(extendedModels, function(m){
  p <- perplexity(m$topicmodel, tfidfData$cleanedTestMatrix)
  p
})
perModelPerplexity
length(perModelPerplexity)
length(topics_number)

perpl_for_testset <- rescale(unlist(perModelPerplexity))

df_perplexity_all <- data.frame(topics_number, perpl_for_testset, random_forest_accuracy_all)
df2_perplexity_all <- melt(data = df_perplexity_all, id.vars = "topics_number")

dev.new()
ggplot(data = df2_perplexity_all, aes(x = topics_number, y = value, linetype=variable)) + 
  geom_line() + labs(x = "Topics number", y="Perplexity and Random Forest Error")

v = TRUE

a <- if(v){
  c(1)
}else{
  c(2)
}
a       

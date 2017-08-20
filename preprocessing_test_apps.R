


setwd('C:/doc/s/sem2/chudy/repo/pub_lda')
source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')

tree_number <- 50
topic_number <- 10

filePath = 'C:/doc/s/sem2/chudy/repo/pub_lda/apps_desc.csv'

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
plot(all_result$topics, all_result$Griffiths2004,type = "l", col="black", xlab = "Topics number", ylab="log(C|K)", main = "Classification performance for dirrerent values of topic number")

dev.new()
plot(all_result$topics, all_result$CaoJuan2009,type = "l", col="black", xlab = "Topics number", 
     ylab="", main = "")

dev.new()
plot(all_result$topics, all_result$Arun2010,type = "l", col="black", xlab = "Topics number", 
     ylab="", main = "")


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

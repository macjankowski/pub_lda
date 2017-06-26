


setwd('C:/doc/s/sem2/chudy/repo/pub_lda')
source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')

tree_number <- 500
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

res_lda_30 <- trainAndPredict(tree_number, lda_30$ldaTrainData, tfidfData$cleanedTrainLabels, 
                           lda_30$ldaTestData, tfidfData$cleanedTestLabels)


plotResults(res_lda_30$testResult$threshold, res_lda_30$testResult$bridgeRatio, res_lda_30$testResult$errorRatio)

res_lda_30$model

######################################### Estimate topic count using LSA ##############################################

res <- estimateTopicsCount(2,1000,5, tfidfData = tfidfData, tree_number = tree_number)

topics = seq(from = 2, to = 1000, by = 5)

errors <- res$rfResult
dev.new()
plot(topics, errors, type = "l", main = "Performance", xlab = "Topics number", 
     ylab = "Error rate on test data", col="black", col.axis = "dimgray", col.lab = "blueviolet")

topics[7]



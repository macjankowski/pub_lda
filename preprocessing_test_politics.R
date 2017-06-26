


setwd('C:/doc/s/sem2/chudy/repo/pub_lda')
source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')

tree_number <- 500
topic_number <- 30

filePath = 'C:/doc/s/sem2/chudy/repo/pub_lda/politiciants data.csv'
tweetsAll <- cleanData(filePath)

labelMapping <- data.frame(message = c("attack", "constituency", "information", "media", "mobilization", "personal", "policy", "support", "other"),
                           label = c(0,1,2,3,4,5,6,7,8))

tweetsAllLabelsNumeric <- labelsToNumeric(tweetsAll, labelMapping)

tweets <- partitionData(tweetsAllLabelsNumeric)
dim(tweets$train)
dim(tweets$test)

politicsData <- tweets

tfData <- prepareTfIdfWithLabels(politicsData)
lda <- calculateLDA(tfData, topic_number)
posterior(lda$topicmodel)[2]

res <- trainAndPredict(tree_number, lda$ldaTrainData, tfData$cleanedTrainLabels, 
                      lda$ldaTestData, tfData$cleanedTestLabels)


plotResults(res$testResult$threshold, res$testResult$bridgeRatio, res$testResult$errorRatio)

res$model

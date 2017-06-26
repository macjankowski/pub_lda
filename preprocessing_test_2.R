


setwd('C:/doc/s/sem2/chudy/repo/pub_lda')
source('./preprocessing_2.R')
source('./dimRed.R')
source('./classification.R')

tree_number <- 500
topic_number <- 30

filePath = 'C:/doc/s/sem2/chudy/repo/pub_lda/politiciants data.csv'
tweetsAll <- cleanData(filePath)

tweetsAllLabelsNumeric <- labelsToNumeric(tweetsAll)

tfIdfWithLabels <- prepareTfIdfWithLabels(tweetsAllLabelsNumeric)

partitioned <- partitionData(tfIdfWithLabels)
dim(partitioned$train)
dim(partitioned$test)


lda <- calculateLDA(partitioned, topic_number)
posterior(lda$topicmodel)[2]

res <- trainAndPredict(tree_number, lda$ldaTrainData, tfData$cleanedTrainLabels, 
                      lda$ldaTestData, tfData$cleanedTestLabels)


plotResults(res$testResult$threshold, res$testResult$bridgeRatio, res$testResult$errorRatio)

res$model

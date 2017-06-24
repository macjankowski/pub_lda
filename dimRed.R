library("topicmodels")

calculateLDA <- function(tfData, topic_number){
  
  topicmodel <- LDA(tfData$cleanedTrainMatrix, k=topic_number, control=list(seed=SEED))
  trainData <- posterior(topicmodel)[2]$topics
  testData <- posterior(topicmodel, tfData$cleanedTestMatrix)[2]$topics
  
  list(topicmodel=topicmodel, ldaTrainData=trainData, ldaTestData=testData)
}
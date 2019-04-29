library(randomForest)

trainAndPredict <- function(tree_number, trainData, trainLabels, testData, testLabels){
  start.time <- Sys.time()
  model <- randomForest(x=as.matrix(trainData), y=trainLabels, ntree=tree_number, keep.forest=TRUE)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  testResult <- test(model, testData, testLabels)
  list(model=model, testResult=testResult, duration=duration)
}

test <- function(model, testMatrix, test_code, threshold = seq(0,1,by=0.01)) {
  
  bridge_for_threshold <- function(threshold){
    
    confidence_value <- apply(out,1, max)
    confidence_value_for_threshold <- confidence_value > threshold
    indices_for_threshold <- which(confidence_value_for_threshold)
    
    if(length(indices_for_threshold) > 1){
      out_for_threshold <- out[indices_for_threshold,]
      test_code_for_threshold <-   test_code[indices_for_threshold]
      
      if(length(indices_for_threshold) == 1){
        best.class.index <- which.max(out_for_threshold)
        best.class <- column.names[best.class.index]
      }else{
        best.class.index <- apply(out_for_threshold,1, which.max)
        best.class <- column.names[best.class.index]
      }
     
      test.result <- best.class[best.class == test_code_for_threshold]
      error <- 1-length(test.result)/length(test_code_for_threshold)
      
      selected_count <- length(indices_for_threshold)
      selected_count_ratio <- selected_count/length(test_code)
      
      list(error_ratio=error*100, bridge_ratio=selected_count_ratio*100, bridge_count=selected_count)
    }else{
      list(error_ratio=0, bridge_ratio=0, bridge_count=0)
    }
  }
  
  out <- predict(model, as.matrix(testMatrix), type="prob")
  column.names <- colnames(out)
  
  bridge <- lapply(threshold, bridge_for_threshold)
  bridge_ratio <- lapply(bridge, function(x){x$bridge_ratio})
  error_ratio <- lapply(bridge, function(x){x$error_ratio})
  
  list(threshold=threshold, bridgeRatio=bridge_ratio, errorRatio=error_ratio)
}

distributionOfConfidence <- function(model, testMatrix){
  out <- predict(model, as.matrix(testMatrix), type="prob")
  confidence_value <- apply(out,1, max)
  confidence_value
}

trainAndPredictSimple <- function(tree_number, trainData, trainLabels, testData, testLabels){
  start.time <- Sys.time()
  model <- randomForest(x=as.matrix(trainData), y=trainLabels, ntree=tree_number, keep.forest=TRUE)
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  out <- as.integer(predict(model, as.matrix(testData)))-1
  test.result <- out[out != testLabels]
  errorRate <- length(test.result)/length(testLabels)
  errorRate
}

predictSimple <- function(model, testData, testLabels){
  p <- predict(model, as.matrix(testData))
  out <- as.integer(p)-1
  test.result <- out[out != testLabels]
  errorRate <- length(test.result)/length(testLabels)
  errorRate
}

predictSimpleLinearDiscriminantAnalysis <- function(model, testData, testLabels){
  out <- as.integer(predict(model, as.matrix(testData))$class)-1
  test.result <- out[out != testLabels]
  errorRate <- length(test.result)/length(testLabels)
  errorRate
}

trainRfOnLda <- function(ldaModel, trainLabels){
  trainData <- ldaModel$ldaTrainData
  testData <- ldaModel$ldaTestData
  rfModel <- randomForest(x=as.matrix(trainData), y=trainLabels, ntree=tree_number, keep.forest=TRUE)
  rfModel
}

trainSVMOnLda <- function(ldaModel, trainLabels){
  trainData <- ldaModel$ldaTrainData
  testData <- ldaModel$ldaTestData
  svmModel <- svm(x=as.matrix(trainData), y=trainLabels)
  svmModel
}

trainNbOnLda <- function(ldaModel, trainLabels){
  trainData <- ldaModel$ldaTrainData
  testData <- ldaModel$ldaTestData
  m <- naive_bayes(x=as.matrix(trainData), y=trainLabels, laplace = 0.000001)
  m
}

trainLinearDiscriminantAnalysisOnLda <- function(ldaModel, trainLabels){
  print(paste("Training LDA for k=",ldaModel$topicmodel@k))
  trainData <- ldaModel$ldaTrainData
  testData <- ldaModel$ldaTestData
  m <- lda(x=as.matrix(trainData), grouping = trainLabels)
  m
}
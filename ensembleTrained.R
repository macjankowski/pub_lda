


lda_models_200_for_ensemble <- lapply(lda_models_200, function(m){
  list(ldaTrainData=m$ldaTrainData[1:1000,], ldaTestData=m$ldaTrainData[1001:1520,])
})

rfModels_200_for_ensemble <- lapply(lda_models_200_for_ensemble, function(m){
  trainRfOnLda(m, trainLabels[1:1000])
})

prediction_for_ensemble <- lapply(1:length(lda_models_200), function(i){
  predictSimple(rfModels_200_for_ensemble[[i]], lda_models_200_for_ensemble[[i]]$ldaTestData, trainLabels[1001:1520])
})

p <- predictWithScoreEnsemble(rfModels_200_for_ensemble, lda_models_200_for_ensemble)

####################################### train rf on classes  #############################################
testLabels
ensClasses <- data.frame(t(p$classes))
trainDataEnsemble <- as.data.frame(lapply(ensClasses, factor))

dim(trainDataEnsemble)

m <-as.numeric(trainLabels[1001:1520])
m[500] <- 23
trainLabelsEnsemble <- factor(m-1)
trainLabelsEnsemble
length(trainLabelsEnsemble)
trainLabelsEnsemble <- trainLabels[1001:1520]
levels(trainLabelsEnsemble) <- 0:24
trainLabelsEnsemble

rfModelEnsemble <- randomForest(x=trainDataEnsemble, y=trainLabelsEnsemble, ntree=500, keep.forest=TRUE, mtry=10)
rfModelEnsemble
p_test <- predictWithScoreEnsemble(rfModels_200, lda_models_200)
errorForEnsembleWithScoreResult(p_test$classes, p_test$scores, testLabels)

ensTestClasses <- data.frame(t(p_test$classes))
testDataEnsemble <- as.data.frame(lapply(ensTestClasses, factor))

predict(rfModelEnsemble, as.matrix(testDataEnsemble))

predictSimple(rfModelEnsemble, testDataEnsemble, testLabels)

####################################### train rf on confidences  #############################################

prediction_for_ensemble_conf <- lapply(1:length(rfModels_200_for_ensemble), function(i){
  predict(rfModels_200_for_ensemble[[i]], lda_models_200_for_ensemble[[i]]$ldaTestData, type="prob")
})

l <- list()

for(i in 1:length(prediction_for_ensemble_conf)){
  l <- cbind(l, prediction_for_ensemble_conf[[i]])
}

m2 <- as.numeric(trainLabelsEnsemble)-1
m2[10] <- 24
trainLabelsEnsemble <- factor(m2)
trainLabelsEnsemble

# train ensemble
rfModelEnsemble_conf <- randomForest(x=l, y=trainLabelsEnsemble, ntree=500, keep.forest=TRUE, mtry = 100)
rfModelEnsemble_conf

####### data for prediction #########

testDataForEnsemble_scores <- lapply(1:length(lda_models_200), function(i){
  predict(rfModels_200[[i]], lda_models_200[[i]]$ldaTestData, type="prob")
})


l2 <- list()

for(i in 1:length(testDataForEnsemble_scores)){
  l2 <- cbind(l2, testDataForEnsemble_scores[[i]])
}

predictSimple(rfModelEnsemble_conf, l2, testLabels)

predict(rfModelEnsemble_conf, as.matrix(l2)) == testLabels


testLabels





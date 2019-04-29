library(stm)

poliblog.documents
(poliblog.ratings+100)/200

writeLdac(poliblog.documents, '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/poliblog.documents.txt')
write((poliblog.ratings+100)/200, file='/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/poliblog.ratings.txt',
      ncolumns=1,sep="\n")

############################## poliblog train/test ##############################



################################ analysys ##########################################

ldasPerClass <- createPerClassLda(
  K=30, 
  trainLabels = reviewsTrainLabels,
  trainTf=partitionedReviewsTF$cleanedTrainMatrix
)

ldasOnly <- lapply(ldasPerClass, function(x){x$topicmodel})
resp <- classifyUsingPerplexity(ldasOnly, partitionedReviewsTF$cleanedTestMatrix)
resp

resp_l <- classifyUsingLikelihood(ldasPerClass, partitionedReviewsTF$cleanedTestMatrix)
resp_l
calculateErrorRate(intTestLabels, resp_l)



intTestLabels <- as.integer(partitionedReviewsTF$cleanedTestLabels)
calculateErrorRate(intTestLabels, resp)

tree_number <- 500
reviewsTrainLabels <- partitionedReviewsTF$cleanedTrainLabels
reviewsTrainLabels
subsetForClass_1 <- which(reviewsTrainLabels==1)

reviews_perClass_topicmodel_100_1 <- LDA(
  partitionedReviewsTF$cleanedTrainMatrix[subsetForClass_1,], 
  k=100, 
  control=list(seed=SEED)
)
reviews_perClass_topicmodel_100_1_train <- posterior(reviews_perClass_topicmodel_100_1)[2]$topics

reviews_perClass_topicmodel_100_1

subsetForClass_0 <- which(reviewsTrainLabels==0)
reviews_perClass_topicmodel_100_0 <- LDA(
  partitionedReviewsTF$cleanedTrainMatrix[subsetForClass_0,], 
  k=100, 
  control=list(seed=SEED)
)
reviews_perClass_topicmodel_100_0_train <- posterior(reviews_perClass_topicmodel_100_0)[2]$topics

reviewsTrainLabels_class_1 <- reviewsTrainLabels[reviewsTrainLabels==1]
reviewsTrainLabels_class_0 <- reviewsTrainLabels[reviewsTrainLabels==0]

reviews_perClass_topicmodel_100_0_1_train <- rbind(reviews_perClass_topicmodel_100_0_train, reviews_perClass_topicmodel_100_1_train)
dim(reviews_perClass_topicmodel_100_0_1)

reviewsTrainLabels_class_0_1 <- factor(c(as.character(reviewsTrainLabels_class_0), as.character(reviewsTrainLabels_class_1)))
reviewsTrainLabels_class_0_1
length(reviewsTrainLabels_class_0_1)

reviews_perClass_rfModel_100 <- randomForest(x=as.matrix(reviews_perClass_topicmodel_100_0_1_train), 
                                             y=reviewsTrainLabels_class_0_1, 
                                             ntree=tree_number, 
                                             keep.forest=TRUE)

############## build test dataset #########################

reviewsTestLabels <- partitionedReviewsTF$cleanedTestLabels
reviewsTestLabels

chosenModelsTmp <- chooseModelUsingPerplexity(partitionedReviewsTF$cleanedTestMatrix, 
                                              list(reviews_perClass_topicmodel_100_0, reviews_perClass_topicmodel_100_1))

perplexity(reviews_perClass_topicmodel_100_0, partitionedReviewsTF$cleanedTestMatrix[1,])
perplexity(reviews_perClass_topicmodel_100_1, partitionedReviewsTF$cleanedTestMatrix[1,])


mmm <- list(reviews_perClass_topicmodel_100_0, reviews_perClass_topicmodel_100_1)
resp <- classifyUsingPerplexity(mmm, partitionedReviewsTF$cleanedTestMatrix)
resp

reviewsTestLabels[1:100]

respXor <- xor(resp, as.integer(reviewsTestLabels))
length(respXor[respXor == TRUE])/length(reviewsTestLabels)

######################################### experiment up to 1000 topics ##############################################

reviewsRange <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000)
reviewsLdaModels <- lapply(reviewsRange, function(x){calculateLDA(tfData = partitionedReviewsTF, topic_number = x)})

reviewsTrainLabels <- partitionedReviewsTF$cleanedTrainLabels
reviewsTestLabels <- partitionedReviewsTF$cleanedTestLabels

reviewsRfModels <- lapply(reviewsLdaModels, function(x){
  trainRfOnLda(x, reviewsTrainLabels)
})

reviewsRfErrors <- sapply(1:length(reviewsRange), function(i){
  predictSimple(reviewsRfModels[[i]], reviewsLdaModels[[i]]$ldaTestData, reviewsTestLabels)
})

reviewsSVMModels <- lapply(reviewsLdaModels, function(x){
  trainSVMOnLda(x, reviewsTrainLabels)
})

reviewsSVMErrors <- sapply(1:length(reviewsRange), function(i){
  predictSimple(reviewsSVMModels[[i]], reviewsLdaModels[[i]]$ldaTestData, reviewsTestLabels)
})

plot(reviewsRange, reviewsSVMErrors, type="l")
plot(reviewsRange, reviewsRfErrors, type="l")

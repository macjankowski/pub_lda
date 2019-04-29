library(VGAM)
library(lm.beta)
library(MASS) 

################## create parameters for synthetic dataset based on movie review dataset ############################

synthV <- 500
synthK <- 100
synthM <- 1386

docLengths <- apply(partitionedReviewsTF$cleanedTrainMatrix,1,function(v){
  sum(v[v != 0])
})
synthLambda <- sum(docLengths)/length(docLengths)
synthLambda

synthAlpha <- reviewsLdaModels[[13]]$topicmodel@alpha
synthAlpha 

testBeta <- exp(reviewsLdaModels[[13]]$topicmodel@beta)[,1:synthV]
testBeta

delta <- diri.est(testBeta)
delta

############################################### Generate Synthetic Dataset ##########################################

docs <- generateCorpus(V = synthV, M = synthM, K = synthK, alpha = synthAlpha, delta = delta$param, lambda = synthLambda)
dim(docs)

synthTrainDocs <- docs[1:1108,]
synthTestDocs <- docs[1109:1386,]

syntheticTf = list(cleanedTrainMatrix=as.DocumentTermMatrix(synthTrainDocs, weighting=weightTf), 
                   cleanedTestMatrix=as.DocumentTermMatrix(synthTestDocs, weighting=weightTf))
syntheticTf

############################################ set up cluster ###########################################
library(parallel)

# Calculate the number of cores
no_cores <- 4
no_cores

# Initiate cluster
cl <- makeCluster(no_cores, type="FORK")

######################################### experiment up to 1000 topics ##############################################

synthetic100Range <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200, 300, 400, 500, 600, 1000)
synthetic100RangeShuffled <- sample(synthetic100Range)
synthetic100RangeShuffled
synthetic100Models <- parLapply(cl, synthetic100RangeShuffled, function(x){calculateLDA(tfData = syntheticTf, topic_number = x)})

stopCluster(cl)

synthetic100LdaModels <- synthetic100Models[order(sapply(synthetic100Models, function(m){
  m$topicmodel@k
}))]

######################################### Average entropy ##############################################

synth100SubRange <- 1:23
avgEntropyForSynthetic100Models <- unlist(lapply(synthetic100LdaModels, avgEntropyForModel))
avgEntropyForSynthetic100Models
plot(synthetic100Range[synth100SubRange], avgEntropyForSynthetic100Models[synth100SubRange], type="l")


avgEntropyForPosteriorForSynthetic100Models <- unlist(lapply(synthetic100LdaModels, avgEntropyForPosterior1))
avgEntropyForPosteriorForSynthetic100Models
plot(synthetic100Range[synth100SubRange], avgEntropyForPosteriorForSynthetic100Models[synth100SubRange], type="l")





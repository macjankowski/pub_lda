library(VGAM)
library(lm.beta)
library(MASS) 
library(Compositional)
library(MCMCpack)
library(tm)

################## create parameters for synthetic dataset based on movie review dataset ############################

synth50V <- 500
synth50K <- 50
synth50M <- 1386
synth50Lambda <- 186
synth50Alpha <- 30

testBeta50 <- exp(reviewsLdaModels[[8]]$topicmodel@beta)[,1:synthV]
testBeta50

delta50 <- diri.est(testBeta50)
length(delta50$param)
delta50$param


############################################### Generate Synthetic Dataset ##########################################

#testDocs50 <- generateCorpus(V = synth50V, M = 5, K = synth50K, alpha = synth50Lambda, delta = delta50$param, lambda = synth50Lambda)

docs50 <- generateCorpus(V = synth50V, M = synth50M, K = synth50K, alpha = synth50Lambda, delta = delta50$param, lambda = synth50Lambda)
dim(docs50)

synthTrainDocs50 <- docs50[1:1108,]
synthTestDocs50 <- docs50[1109:1386,]

syntheticTf50 = list(cleanedTrainMatrix=as.DocumentTermMatrix(synthTrainDocs50, weighting=weightTf), 
                   cleanedTestMatrix=as.DocumentTermMatrix(synthTestDocs50, weighting=weightTf))
syntheticTf50

############################################ set up cluster ###########################################
library(parallel)

# Calculate the number of cores
no_cores <- 4
no_cores

# Initiate cluster
cl <- makeCluster(no_cores, type="FORK")

######################################### experiment up to 1000 topics ##############################################

synthetic50Range <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
synthetic50RangeShuffled <- sample(synthetic50Range)
synthetic50RangeShuffled
synthetic50Models <- parLapply(cl, synthetic50RangeShuffled, function(x){calculateLDA(tfData = syntheticTf50, topic_number = x)})

stopCluster(cl)

synthetic50LdaModels <- synthetic50Models[order(sapply(synthetic50Models, function(m){
  m$topicmodel@k
}))]

######################################### Analyse entropy entropy 100 topics ##############################################

synth100SubRange <- 1:23
avgEntropyForSynthetic100Models <- unlist(lapply(synthetic100LdaModels, avgEntropyForModel))
avgEntropyForSynthetic100Models
plot(synthetic100Range[synth100SubRange], avgEntropyForSynthetic100Models[synth100SubRange], type="l")


avgEntropyForPosteriorForSynthetic100Models <- unlist(lapply(synthetic100LdaModels, avgEntropyForPosterior1))
avgEntropyForPosteriorForSynthetic100Models
plot(synthetic100Range[synth100SubRange], avgEntropyForPosteriorForSynthetic100Models[synth100SubRange], type="l")


######################################### Analyse entropy entropy 50 topics ##############################################

synth50SubRange <- 1:23
avgEntropyForSynthetic50Models <- unlist(lapply(synthetic50LdaModels, avgEntropyForModel))
avgEntropyForSynthetic50Models
plot(synthetic50Range[synth50SubRange], avgEntropyForSynthetic50Models[synth50SubRange], type="l")


avgEntropyForPosteriorForSynthetic50Models <- unlist(lapply(synthetic50LdaModels, avgEntropyForPosterior1))
avgEntropyForPosteriorForSynthetic50Models
plot(synthetic50Range[synth50SubRange], avgEntropyForPosteriorForSynthetic50Models[synth50SubRange], type="l")

##################### analyse 4 methods #########################

synthetic50LdaTuning <- estimateTopicsCount4MethodsRange(synthetic50Range, tfidfData = syntheticTf50)

plotFourMethods(synthetic50Range, synthetic50LdaTuning)

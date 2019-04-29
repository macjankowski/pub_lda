library(VGAM)
library(lm.beta)
library(MASS) 
library(Compositional)
library(MCMCpack)
library(tm)
library(parallel)


################## create parameters for synthetic dataset based on movie review dataset ############################

synth50VSmall <- 30
synth50KSmall <- 50
synth50MSmall <- 800
synth50LambdaSmall <- 186
synth50AlphaSmall <- 30

testBeta50Small <- exp(reviewsLdaModels[[8]]$topicmodel@beta)[,1:synth50VSmall]
testBeta50Small

synth50deltaSmall = 0.5



############################################### Generate Synthetic Dataset ##########################################

#testDocs50 <- generateCorpus(V = synth50V, M = 5, K = synth50K, alpha = synth50Lambda, delta = delta50$param, lambda = synth50Lambda)

docs50Small <- generateCorpus(V = synth50VSmall, M = 10, K = synth50KSmall, alpha = synth50AlphaSmall, 
                         delta = synth50deltaSmall, lambda = synth50LambdaSmall)
dim(docs50Small)

synthTrainDocs50Small <- docs50Small[1:600,]
synthTestDocs50Small <- docs50Small[600:800,]

syntheticTf50Small = list(cleanedTrainMatrix=as.DocumentTermMatrix(synthTrainDocs50Small, weighting=weightTf), 
                     cleanedTestMatrix=as.DocumentTermMatrix(synthTestDocs50Small, weighting=weightTf))
syntheticTf50Small

######################################### experiment up to 1000 topics ##############################################

synthetic50RangeSmall <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
synthetic50ModelsSmall <- lapply(synthetic50RangeSmall,function(x){calculateLDA(tfData = syntheticTf50Small, topic_number = x)})

stopCluster(cl)

synthetic50LdaModelsSmall <- synthetic50ModelsSmall[order(sapply(synthetic50ModelsSmall, function(m){
  m$topicmodel@k
}))]

synthetic50RangeSmall[[1]]$topicmode

######################################### experiment up to 1000 topics ##############################################

synth50SubRangeSmall <- 1:23

avgEntropyForPosteriorForSynthetic50ModelsSmall <- unlist(lapply(synthetic50LdaModelsSmall, avgEntropyForPosterior1))
avgEntropyForPosteriorForSynthetic50ModelsSmall
plot(synthetic50RangeSmall[synth50SubRangeSmall], avgEntropyForPosteriorForSynthetic50ModelsSmall[synth50SubRangeSmall], type="l")

alphas50 <- sapply(synthetic50LdaModelsSmall, function(m){m$topicmodel@alpha})

dev.new()
plot(synthetic50RangeSmall, alphas50, type="l")


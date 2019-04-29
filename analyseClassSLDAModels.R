library(scales)
library(reshape)
library(ggplot2)

############################################## analyse poliblog from class sLDA #################################################

trueLabelsPath='/Users/mjankowski/doc/workspace/blei/unmodified/class-slda/sample-data/poliblog/poliblog.ratings.txt'
trueLabels <- loadLabels(file=trueLabelsPath)
poliblogRange <- c(2,5,10,15,20,30,50,75,100,150,200,300,400,500)

poliblogSubrange = 1:length(poliblogRange)


mm <- loadClassSLDAPosterior(path='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_10_inf',
                             truelabelsPath = trueLabelsPath)
mm

politicalBlogInference <- readPrediction(rootPath='/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_poliblog_infer_single_2-500',
                      topicCounts = poliblogRange,
                      truelabelsPath = trueLabelsPath, withProbs = FALSE)
politicalBlogInference

poliblogErrorsUnscaled <- sapply(politicalBlogInference, function(m){m$errorRate})

plot(poliblogRange, poliblogErrorsUnscaled, type="l")

analyseMutualInformationSLDA(poliblogRange[poliblogSubrange], politicalBlogInference[poliblogSubrange], trueLabels, bins=2, header="Poliblog, bins=2")

analyseConditionalEntropySLDA(poliblogRange[poliblogSubrange], politicalBlogInference[poliblogSubrange], trueLabels, bins=2)

###################################################### second analysis #########################################################

############################################## analyse sms from class sLDA #################################################

trueLabelsPathSms='/Users/mjankowski/doc/workspace/blei/unmodified/class-slda/sample-data/smsSpam/smsSpamTest.isSpam.txt'
trueLabelsSms <- loadLabels(file=trueLabelsPathSms)
smsRange <- c(5,10,15,20,30,40,50,75,100)

smsSubrange = 1:length(smsRange)


smsSpamInference <- readPrediction(rootPath='/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_sms_infer_single_5-100',
                                         topicCounts = smsRange,
                                         truelabelsPath = trueLabelsPathSms,
                                          withProbs = FALSE)

smsSpamErrorsUnscaled <- sapply(smsSpamInference, function(m){m$errorRate})

plot(smsRange, smsSpamErrorsUnscaled, type="l")

analyseMutualInformationSLDA(smsRange[smsSubrange], smsSpamInference[smsSubrange], trueLabelsSms, bins=2, header="Sms Spam, bins=2")

analyseConditionalEntropySLDA(poliblogRange[poliblogSubrange], politicalBlogInference[poliblogSubrange], trueLabels, bins=2)

###################################################### second analysis #########################################################

smsTrueLabelsPath <- '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpam.isSpam.txt'
smsTrueLabels <- loadLabels(file=smsTrueLabelsPath)


boostWithProbs <- readPrediction(rootPath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_inference_10_iters',
               topicCounts = 0:9,
               truelabelsPath = smsTrueLabelsPath)

L <- length(boostWithProbs)
L
M <- length(boostWithProbs[[1]]$probabilities[,1])
M
C <- length(boostWithProbs[[1]]$probabilities[1,])
C
############################### highest prob (without weights) ##############################################

calculateErrorRateForEnsemble(boostWithProbs, rep(1,L))

boostWithProbs[[1]]$errorRate
min(sapply(boostWithProbs, function(m){m$errorRate}))


############################## weighted ###########################################

model_weights <- loadModelsWeights('/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_ensemble_10_iters',0:9)
model_weights

calculateErrorRateForEnsemble(boostWithProbs, model_weights)

############################# ada boost classical approach ####################################

boostWithProbs <- readPrediction(rootPath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_inference_10_iters',
                                 topicCounts = 0:9,
                                 truelabelsPath = smsTrueLabelsPath)

range = 1:10

iters <- unlist(lapply(range, function(i){
  err <- calculateErrorRateForEnsembleAdaBoostSubset(boostWithProbs, model_weights, i)
  err
}))



boosting_iterations <- sapply(boostWithProbs, function(m){m$errorRate})

length(range)
length(boosting_iterations)
length(iters)

boostClassificationError <- data.frame(range, boosting_iterations, iters)
boostClassificationErrorsMelted <- melt(data = boostClassificationError, id.vars = "range")

dev.new()
ggplot(data = boostClassificationErrorsMelted, aes(x = range, y = value, 
  color = factor(variable, labels = c("Single model error", "Ensemble error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Boosting iterations", y="Test Error", color="Variable") + theme_classic(base_size = 18)+
  ggtitle("AdaBoost for Single LDA Model")


############################# ada boost classical approach ####################################

range = 1:20

boostWithProbs2 <- readPrediction(rootPath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_inference_10_iters',
                                 topicCounts = range-1,
                                 truelabelsPath = smsTrueLabelsPath)


model_weights <- loadModelsWeights('/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_ensemble_10_iters',range-1)
model_weights

iters <- unlist(lapply(range, function(i){
  err <- calculateErrorRateForEnsembleAdaBoostSubset(boostWithProbs2, model_weights, i)
  err
}))
iters


boosting_iterations <- sapply(boostWithProbs2, function(m){m$errorRate})

length(range)
length(boosting_iterations)
length(iters)

boostClassificationError <- data.frame(range, boosting_iterations, iters)
boostClassificationErrorsMelted <- melt(data = boostClassificationError, id.vars = "range")

dev.new()
ggplot(data = boostClassificationErrorsMelted, aes(x = range, y = value, 
                                                   color = factor(variable, labels = c("Single model error", "Ensemble error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Boosting iterations", y="Test Error", color="Variable") + theme_classic(base_size = 18)+
  ggtitle("AdaBoost for Single LDA Model")

smsTrueLabelsPathForTest <- '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpamTest.isSpam.txt'
smsTrueLabelsForTest <- loadLabels(file=smsTrueLabelsPathForTest)
length(smsTrueLabelsForTest)
smsTrueLabelsForTest

smsTrueLabelsPathForTrain <- '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpam.isSpam.txt'
smsTrueLabelsForTrain <- loadLabels(file=smsTrueLabelsPathForTrain)
length(smsTrueLabelsForTrain)
smsTrueLabelsForTrain

####################### 20 topics 20 iters ############################
analyseBoosting(range=range, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_inference_20_iters_on_tests', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_ensemble_10_iters',
                trueLabelsPath=smsTrueLabelsPathForTest,
                header="")

######################## 20 topics 10 iters #######################################
analyseBoosting(range=1:10, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_inference_20_topics_10_iters_on_tests', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_ensemble_20_topics_10_iters',
                trueLabelsPath=smsTrueLabelsPathForTest,
                header="")

######################## sms 20 topics 30 iters #######################################
plot1 <- analyseBoosting(range=1:30, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_inference_20_topics_30_iters_on_tests', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_ensemble_20_topics_30_iters',
                trueLabelsPath=smsTrueLabelsPathForTest,
                header="")

######################## sms 15 topics 100 iters #######################################
plot1 <- analyseBoosting(range=1:50, 
                         rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_inference_15_topics_100_iters', 
                         rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_ensemble_15_topics_100_iters',
                         trueLabelsPath=smsTrueLabelsPathForTest,
                         header="")

######################## poliblog 20 topics 10 iters (models from scratch) #######################################
analyseBoosting(range=1:10, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_inference_10_topics_10_iters_shuffled', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_10_topics_10_iters_shuffled/',
                trueLabelsPath=poliblogTrueTestLabelsPath,
                header="")

######################## poliblog 20 topics 30 iters (models from scratch) #######################################
plot3 <- analyseBoosting(range=1:14, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_inference_10_topics_30_iters_shuffled', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_10_topics_30_iters_shuffled/',
                trueLabelsPath=poliblogTrueTestLabelsPath,
                header="")

######################## poliblog 15 topics 100 iters (models from scratch) #######################################
plot3 <- analyseBoosting(range=1:100, 
                         rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_inference_15_topics_100_iters', 
                         rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_15_topics_100_iters',
                         trueLabelsPath=poliblogTrueTestLabelsPath,
                         header="")

################## combine plots #######################
library(cowplot)
dev.new()
plot_grid(plot1, plot3, labels = "AUTO")

######################## sms 20 topics 15 iters (models from scratch) #######################################
analyseBoosting(range=1:15, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_inference_20_topics_15_iters_on_tests', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/out_sms_ensemble_20_topics_15_iters',
                trueLabelsPath=smsTrueLabelsPathForTest,
                header="")

#################################### analyse singl model poliblog ###########################

poliblog_single_vs_ensemble <- analyseModelForDifferentK(topicNumbers=c(5,10,15,20,30,40,50,75,100), 
  rootInferencePath=     '/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_poliblog_infer_5-100', 
  rootInferenceTrainPath='/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_poliblog_infer_train_5-100',
  trueTestLabelsPath=poliblogTrueTestLabelsPath,
  trueTrainLabelsPath=poliblogTrueTrainLabelsPath,
  header="", ensembleError = 0.22)


################################ analyse single models on smsSpam #########################

sms_single_vs_ensemble <- analyseModelForDifferentK(topicNumbers=c(5,10,15,20,30,40,50,75,100), 
                          rootInferencePath=     '/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_sms_infer_single_5-100', 
                          rootInferenceTrainPath='/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_sms_infer_single_train_5-100',
                          trueTestLabelsPath=smsTrueLabelsPathForTest,
                          trueTrainLabelsPath=smsTrueLabelsPathForTrain,
                          header="", ensembleError = 0.12)

dev.new()
plot_grid(sms_single_vs_ensemble, poliblog_single_vs_ensemble, labels = "AUTO")






smsSpamPredictions <- readPrediction(
  rootPath='/Users/mjankowski/doc/workspace/blei/unmodified/for_inference/class-slda/out_sms_infer_single_5-100',
  topicCounts=c(5,10,15,20,30,40,50,75,100),
  truelabelsPath=smsTrueLabelsPathForTest,
  withProbs=FALSE
)

errsPred <- sapply(smsSpamPredictions, function(p){
  p$errorRate
})

range_single <- c(5,10,15,20,30,40,50,75,100)
plot(range_single, errsPred, type="l")

ens_error <- sapply(errsPred, function(x){0.135})

aaaDf <- data.frame(range_single, errsPred, ens_error)
aaaDfMelted <- melt(data = aaaDf, id.vars = "range_single")

dev.new()
ggplot(data = aaaDfMelted, aes(x = range_single, y = value, 
  color = factor(variable, labels = c("Single model error", "Ensemble error")))) + 
  geom_point() + geom_line(size=1) + 
  labs(x = "Topic number", y="Test Error", color="Variable") + theme_bw()  + #theme_classic(base_size = 18)+
  ggtitle("") + ylim(0, 0.5)

######################## 5-30 topics 10 iters each #######################################

subfolders = c(sapply(0:9, function(x){
  sapply(0:4, function(y){
    paste(x,'/',y, sep = "")
  })
}))

subfolders

range=1:50

smsRootInferPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_inference_5-30_topics_10_iters_each_on_tests'
inference <- readPrediction(rootPath=smsRootInferPath,
                            topicCounts = range,
                            truelabelsPath = smsTrueLabelsPathForTest)
inference

############################################## sms spam 5-30 topics, 10 iters each ########################################################

sms_5_30_plot <- analyseBoosting(range=1:38, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_inference_5-30_topics_10_iters_each_on_tests', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_ensemble_5-30_topics_10_iters_each_copy/',
                trueLabelsPath=smsTrueLabelsPathForTest,
                header="")

analyseBoosting(range=1:25, 
                                 rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_inference_5-30_topics_30_iters_each_on_tests', 
                                 rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_sms_ensemble_5-30_topics_30_iters_each/',
                                 trueLabelsPath=smsTrueLabelsPathForTest,
                                 header="")

############################################## poliblog 5-30 topics, 30 iters each ########################################################

poliblog_5_30_plot <- analyseBoosting(range=1:62, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_infer_5-30_topics_30_iters_shuffled', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_5-30_topics_30_iters_shuffled/',
                trueLabelsPath=poliblogTrueTestLabelsPath,
                header="")

######################## poliblog vs sms ########################

dev.new()
plot_grid(sms_5_30_plot, poliblog_5_30_plot, labels = "AUTO")

####################### poliblog #######################

poliblogTrueTestLabelsPath = '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.ratings.test.txt'
poliblogTrueTestLabels <- loadLabels(file=poliblogTrueTestLabelsPath)
poliblogTrueTestLabels

poliblogTrueTrainLabelsPath = '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.ratings.train.txt'
poliblogTrueTrainLabels <- loadLabels(file=poliblogTrueTrainLabelsPath)
poliblogTrueTrainLabels

############################################## poliblog 5-30 topics, 20 iters each ########################################################


analyseBoosting(range=1:100, 
                rootInferencePath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_inference_5-30_topics_20_iters_each_2', 
                rootModelWeightsPath='/Users/mjankowski/doc/workspace/blei/class-slda/results/out_poliblog_ensemble_5-30_topics_20_iters_each_2/',
                trueLabelsPath=smsTrueLabelsPathForTest,
                header="")

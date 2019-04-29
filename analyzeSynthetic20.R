
################## create parameters for synthetic dataset based on movie review dataset ############################

synth20V <- 100
synth20K <- 20
synth20M <- 800
synth20Lambda <- 186
synth20Alpha <- 30

testBeta20 <- exp(appsLdaModels[[5]]$topicmodel@beta)[,1:synth20V]
testBeta20

synth20DeltaDirichlet <- diri.est(testBeta20)
length(synth20DeltaDirichlet$param)
synth20Delta <- synth20DeltaDirichlet$param
is.vector(synth20Delta)
is.scalar(synth20Delta)

################################## corpus ###############################

docs20 <- generateCorpus(V = synth20V, M = synth20M, K = synth20K, alpha = synth20Alpha, 
                              delta = synth20Delta, lambda = synth20Lambda)
dim(docs20)

synthTrainDocs20 <- docs20[1:600,]
synthTestDocs20 <- docs20[600:800,]

syntheticTf20 = list(cleanedTrainMatrix=as.DocumentTermMatrix(synthTrainDocs20, weighting=weightTf), 
                          cleanedTestMatrix=as.DocumentTermMatrix(synthTestDocs20, weighting=weightTf))
syntheticTf20

synthTrainDocs20

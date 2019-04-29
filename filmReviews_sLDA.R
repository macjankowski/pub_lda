
library(lda)

help(lda)


########################################### poliblog ########################################
demo(slda)
length(poliblog.documents)
length(poliblog.documents)

poliblog.documents[1:10]
dim(poliblog.documents[[1]])
attributes(poliblog.documents[[1]])

class(poliblog.documents[[1]][1,1])

######################################### movie reviews ##############################################

M = dim(partitionedReviewsTF$cleanedTrainMatrix)[1]
V = dim(partitionedReviewsTF$cleanedTrainMatrix)[2]

reviewsTrainMatrix <- as.matrix(partitionedReviewsTF$cleanedTrainMatrix)

reviewsSLda.vocab <- attributes(reviewsTrainMatrix)$dimnames$Terms
reviewsSLda.documents <- toLdaFormat(partitionedReviewsTF$cleanedTrainMatrix)

#reviewsTrainLabelsNumeric <- as.numeric(levels(reviewsTrainLabels))[reviewsTrainLabels]
reviewsSLda.labels <- reviewsTrainLabels
levels(reviewsSLda.labels) <- c(-1,1)
reviewsSLda.labels <- as.numeric(levels(reviewsSLda.labels))[reviewsSLda.labels]

params <- sample(c(-1, 1), 10, replace=TRUE)
attributes(reviewsSLda.documents[[1]])
reviewsSLda.documents <- as.integer(reviewsSLda.documents)
class(reviewsSLda.documents[[1]][1,1])



result <- slda.em(documents=reviewsSLda.documents,
                  K=10,
                  vocab=reviewsSLda.vocab,
                  num.e.iterations=10,
                  num.m.iterations=4,
                  alpha=1.0, eta=0.1,
                  annotations = reviewsSLda.labels,
                  params = params,
                  variance=0.25,
                  lambda=1.0,
                  logistic=FALSE,
                  method="sLDA")


reviewsThetas <- t(sapply(result$assignments, function(doc){
  tab <- tabulate(doc+1, nbins=10)
  tab/length(doc)
}))

dim(reviewsThetas)

dim(result$topics)
length(result$topic_sums)
as.numeric(result$topic_sums)

reviewsBetas <- result$topics /as.numeric(result$topic_sums)
dim(reviewsBetas)
rowSums(reviewsBetas)

ssss <- generateCorpusFromParameters(K=10, V=V, M=M, thetas=reviewsThetas, beta=reviewsBetas, docLengths)
dim(ssss)
reviewsSLdaObservedCountsFromSample <- observedCountsFromCorpus(ssss)

length(reviewsSLdaObservedCountsFromSample)
t <- chisq.test(reviewsSLdaObservedCountsFromSample, p = reviewsDataMultinomialDistribution)

t$statistic
t$p.value
t$parameter

reviewsSldaSampleClass_0 <- ssss[subsetForClass_0,]
reviewsSLdaObservedCountsFromSampleClass_0 <- observedCountsFromCorpus(reviewsSldaSampleClass_0)
reviewsDataMultinomialDistributionClass_0 <- estimateMultinomialFromCorpus(reviewsOriginalTrainMatrix[subsetForClass_0, ])


t2 <- chisq.test(reviewsSLdaObservedCountsFromSampleClass_0, p = reviewsDataMultinomialDistributionClass_0)

t2$statistic
t2$p.value
t2$parameter

t3 <- chisq.test(reviewsSLdaObservedCountsFromSampleClass_0, p = reviewsDataMultinomialDistribution)

t3$statistic
t3$p.value
t3$parameter


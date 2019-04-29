library("topicmodels")
library("lda")
library("tm")

source('./classification.R')
source("./preprocessing.R")

tree_number <- 2000

data(poliblog.documents)
data(poliblog.vocab)
length(poliblog.vocab)


length(poliblog.documents[[1]])
poliblog.documents[[1]]

min(sapply(poliblog.documents, length))

perm <- sample(1:773)
perm
poliblog.documents.shuffled <- poliblog.documents[perm]
poliblog.ratings.shuffled <- poliblog.ratings[perm]

trainSeq <- 1:573
testSeq <- 574:773
length(trainSeq)
length(testSeq)

  
poliblog.documents.train <- poliblog.documents.shuffled[trainSeq]
poliblog.documents.test <- poliblog.documents.shuffled[testSeq]

poliblog.ratings.train <- poliblog.ratings.shuffled[trainSeq]
poliblog.ratings.test <- poliblog.ratings.shuffled[testSeq]

length(poliblog.documents.train)
length(poliblog.documents.test)
length(poliblog.ratings.train)
length(poliblog.ratings.test)

writeLdac(poliblog.documents.train,          '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.documents.train.txt')
write((poliblog.ratings.train+100)/200, file='/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.ratings.train.txt',
      ncolumns=1,sep="\n")

writeLdac(poliblog.documents.test,           '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.documents.test.txt')
write((poliblog.ratings.test+100)/200, file= '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.ratings.test.txt',
      ncolumns=1,sep="\n")




dim(poliblog.documents[[1]])[2]

poliblog.documents[[1]][2,]

poliblog.train.df[1,poliblog.documents[[1]][1,]] <- poliblog.documents[[1]][2,]
poliblog.train.df[1,poliblog.documents[[1]][1,]] 

# poliblog.train.df <- data.frame(matrix(0, ncol = length(poliblog.vocab), nrow = poliblog_docs_num))
# dim(poliblog.train.df)
# for(ind in 1:poliblog_docs_num){
#   poliblog_orig_doc <- poliblog.documents[[ind]]
#   idxs <- poliblog_orig_doc[1,] + 1
#   poliblog.train.df[ind,idxs] <- poliblog_orig_doc[2,]
# }

poliblog_docs_num <- 773

# convert_to_tfIdf <- function(dataset, vocab_length){
#   docs_num <- length(dataset)
#   dataset.df <- data.frame(matrix(0, ncol = vocab_length, nrow = docs_num))
#   for(ind in 1:docs_num){
#     orig_doc <- dataset[[ind]]
#     idxs <- orig_doc[1,] + 1
#     print(idxs)
#     print(orig_doc[2,])
#     dataset.df[ind,idxs] <- orig_doc[2,]
#   }
#   dataset.df
# }

length(poliblog.documents.train)

#poliblog_tf_data$cleanedTrainMatrix

poli_train <- convertPoliblogToTfIdf(poliblog.documents.train, vocab_length=length(poliblog.vocab))
poli_train_labels <- as.factor((poliblog.ratings.train+100)/200)

poli_test <- convertPoliblogToTfIdf(poliblog.documents.test, vocab_length=length(poliblog.vocab))
poli_test_labels <- as.factor((poliblog.ratings.test+100)/200)

dim(as.matrix(poli_train))

poli_train_raw <- convertToTfIdf_raw(poliblog.documents.train, vocab_length=length(poliblog.vocab))
colnames(poli_train_raw) <- poliblog.vocab
write.csv(poli_train_raw, file= '/Users/mjankowski/doc/data/poliblog/for_python/poliblog.data.train.csv', row.names = FALSE)
write((poliblog.ratings.train+100)/200, file='/Users/mjankowski/doc/data/poliblog/for_python/poliblog.ratings.train.csv',
      ncolumns=1,sep="\n")

poli_test_raw <- convertToTfIdf_raw(poliblog.documents.test, vocab_length=length(poliblog.vocab))
colnames(poli_test_raw) <- poliblog.vocab
write.csv(poli_test_raw, file= '/Users/mjankowski/doc/data/poliblog/for_python/poliblog.data.test.csv', row.names = FALSE)
write((poliblog.ratings.test+100)/200, file= '/Users/mjankowski/doc/data/poliblog/for_python/poliblog.ratings.test.csv',
      ncolumns=1,sep="\n")



help(write.csv)

poliblog_data_for_enemble <- list(cleanedTrainMatrix=poli_train, cleanedTrainLabels=poli_train_labels,
     cleanedTestMatrix=poli_test, cleanedTestLabels=poli_test_labels, duration=NULL)

poliblogRange <- c(2,5,10, 15, 20, 30, 40, 50, 60, 75, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200)
poliblogModels <- lapply(poliblogRange, function(x){calculateLDA(tfData = poliblog_data_for_enemble, topic_number = x)})
poliblogLDAModels <- poliblogModels

poli_lda_12345 <- calculateLDA(tfData = poliblog_data_for_enemble, topic_number = 10)

perplexity(poli_lda_12345$topicmodel, poliblog_data_for_enemble$cleanedTestMatrix[4,])

perplexity(poli_lda_12345, document)



chooseModelUsingPerplexityPoli <- function(testDocumentTermMatrix, ldaModels){
  
  nRows <-dim(testDocumentTermMatrix)[1]
  chosenModels <- rep(0,nRows)
  for(i in 1:nRows){
    document <- testDocumentTermMatrix[i,]
    perplexities_for_doc <- lapply(ldaModels, function(m){
      p <- perplexity(m$topicmodel, document)
      p
    })
    idx <- which.min(perplexities_for_doc)
    print(paste('idx = ',idx,', processed ',i,'/',nRows))
    chosenModels[i] <- idx
  }
  chosenModels
}



poli_rfModels_200_5000 <- lapply(poliblogModels, function(x){
  trainRfOnLda(x, poli_train_labels, tree_number=5000)
})

poli_rfModels_200_2000 <- lapply(poliblogModels, function(x){
  trainRfOnLda(x, poli_train_labels, tree_number=2000)
})

poli_rfModels_200_1000 <- lapply(poliblogModels, function(x){
  trainRfOnLda(x, poli_train_labels, tree_number=1000)
})

poli_rfModels_200 <- poli_rfModels_200_1000

poli_accuracies_200 <- sapply(1:length(poliblogRange), function(i){
  predictSimple(poli_rfModels_200[[i]], poliblogModels[[i]]$ldaTestData, poli_test_labels)
})
poli_accuracies_200

#poliblogPerplexityModels <- chooseModelUsingPerplexityPoli(poliblog_data_for_enemble$cleanedTestMatrix, poliblogModels)  
#hist(poliblogPerplexityModels)



#poli_subRange <- which(poli_accuracies_200 <= 0.325)
#poli_subRange
#poli_subRange <- c(5, 7, 8,  9, 10, 12, 19, 20, 22, 23) #which(poli_accuracies_200 <= 0.325) # 5  6  7  8  9 10 12 19 20 22 23
poli_subRange <- which(poli_accuracies_200 <= 0.325)
poliblogRange[poli_subRange]
poliblogPerplexityModels <- chooseModelUsingPerplexityPoli(poliblog_data_for_enemble$cleanedTestMatrix, poliblogModels[poli_subRange])  

cbind(poliblogRange[poli_subRange], poli_accuracies_200[poli_subRange])

min(poli_accuracies_200[poli_subRange])
max(poli_accuracies_200[poli_subRange])



sub_poliblogModels = poliblogModels[poli_subRange]
sub_poli_rfModels_200 = poli_rfModels_200[poli_subRange]

#poli_lda_test_data <-  poliblogLDAModels[[1]]$ldaTestData
poli_error_perplexity <- errorPerplexityEnsemble(poliblogPerplexityModels, poliblogModels[poli_subRange], poli_rfModels_200[poli_subRange], poli_test_labels)
poli_error_perplexity


############################################## Majority Voting ############################################

res <- predictEnsemble(poli_rfModels_200[poli_subRange], poliblogLDAModels[poli_subRange])
errorForEnsembleResult(res, poli_test_labels)

############################################## Majority Voting with Score ############################################


res <- predictWithScoreEnsemble(poli_rfModels_200[poli_subRange], poliblogLDAModels[poli_subRange])
errorForEnsembleWithScoreResult(res$classes, res$scores, poli_test_labels)








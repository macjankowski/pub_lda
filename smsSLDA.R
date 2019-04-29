
library(stm)

smsTrainMatrix <- as.matrix(partitionedSmsSpamTF$cleanedTrainMatrix)
dim(smsTrainMatrix)
smsTrainSLDA.data <- toLdaFormat(partitionedSmsSpamTF$cleanedTrainMatrix)

smsTrainSLDA.vocab <- attributes(smsTrainMatrix)$dimnames$Terms
smsTrainSLDA.labels <- partitionedSmsSpamTF$cleanedTrainLabels

oneBasedVocabulary <- lapply(smsTrainSLDA.data, function(x){x+1})
oneBasedVocabulary[1:5]
writeLdac(oneBasedVocabulary,    '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpam.documents.txt')
write(as.numeric(smsTrainSLDA.labels)-1, file='/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpam.isSpam.txt',
      ncolumns=1,sep="\n")

############################## poliblog train/test ##############################




################### test data #################

smsTestMatrix <- as.matrix(partitionedSmsSpamTF$cleanedTestMatrix)
dim(smsTestMatrix)
smsTestSLDA.data <- toLdaFormat(partitionedSmsSpamTF$cleanedTestMatrix)

smsTestSLDA.vocab <- attributes(smsTestMatrix)$dimnames$Terms
smsTestSLDA.labels <- partitionedSmsSpamTF$cleanedTestLabels

oneBasedVocabularyTest <- lapply(smsTestSLDA.data, function(x){x+1})
oneBasedVocabularyTest[1:5]
writeLdac(oneBasedVocabularyTest,    '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpamTest.documents.txt')
write(as.numeric(smsTestSLDA.labels)-1, file='/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/smsSpam/smsSpamTest.isSpam.txt',
      ncolumns=1,sep="\n")

  
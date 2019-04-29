
library(entropy)    
library (lda)
library(MASS)
library(scales)
library(ggplot2)
library(randomForest)

source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')
source('./ensemble.R')

tree_number <- 500

poliblog_train = '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.documents.train.txt'
poliblog_train_labels = '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.ratings.train.txt'

poliblog_test = '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.documents.test.txt'
poliblog_test_labels = '/Users/mjankowski/doc/workspace/blei/class-slda/sample-data/poliblog/shuffled/poliblog.ratings.test.txt'

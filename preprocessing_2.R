library("tm")
library(tidytext)


cleanData <- function(filePath) {
  data <- read.csv(filePath, sep=";")  
  dim(data)
  
  #convert all text to lower case
  data$text <- tolower(data$text)
  
  # Replace blank space (“rt”)
  data$text <- gsub("rt", "", data$text)
  
  # Replace @UserName
  data$text <- gsub("@\\w+", "", data$text)
  
  # Remove punctuation
  data$text <- gsub("[[:punct:]]", "", data$text)
  
  # Remove links
  data$text <- gsub("http\\w+", "", data$text)
  
  # Remove tabs
  data$text <- gsub("[ |\t]{2,}", "", data$text)
  
  # Remove blank spaces at the beginning
  data$text <- gsub("^ ", "", data$text)
  
  # Remove blank spaces at the end
  data$text <- gsub(" $", "", data$text)
  
  
  data
}

labelsToNumeric <- function(data){
  labelMapping <- data.frame(message = c("attack", "constituency", "information", "media", "mobilization", "personal", "policy", "support", "other"),
                             label = c(0,1,2,3,4,5,6,7,8))
  
  shuffledWithNumericLabels <- merge(data, labelMapping, by.x = "message", by.y = "message")[sample(nrow(data)),]
  
  shuffledWithNumericLabels[,c(3,2)]
}

prepareTfIdfWithLabels <- function(data, sparseLevel=.998, ngramCount = 1){
  
  start.time <- Sys.time()
  code <- as.factor(data$label)
  
  tf_matrix <- create_matrix(data$text, language="english", removeNumbers=FALSE, stemWords=FALSE, 
                             removeSparseTerms=sparseLevel, ngramLength=ngramLength)
  
  dim(tf_matrix)
  
  rowTotals <- apply(tf_matrix, 1, sum) #Find the sum of words in each Document
  cleanedTfMatrix   <- tf_matrix[rowTotals> 0, ]
  cleanedLabels <- code[rowTotals> 0]
  
  tfIdfMatrix=weightTfIdf(cleanedTfMatrix)

  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken

  list(tfIdfMatrix=tfIdfMatrix, labels=cleanedLabels, duration=duration)
}

partitionData <- function(data){
  
  tfidf <- data$tfIdfMatrix
  tfIdfAsDf <- tidy(tfidf)
  
  dfData <- data$labels, 
  
  ## 75% of the sample size
  smp_size <- floor(0.80 * nrow(data))
  
  ## set the seed to make your partition reproductible
  set.seed(123)
  train_ind <- sample(seq_len(nrow(data)), size = smp_size)
  
  train <- data[train_ind, ]
  test <- data[-train_ind, ]
  
  list(train = train, test = test)
}





createTfIdfMatrix <- function(data, sparseLevel=0,  ngramLength = 1) {
  start.time <- Sys.time()

  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(rawTrainMatrix=train_doc_matrix, rawTestMatrix=test_doc_matrix, duration=duration)
}

create_matrix <- function (textColumns, language = "english", minDocFreq = 1, 
                           maxDocFreq = Inf, minWordLength = 3, maxWordLength = Inf, 
                           ngramLength = 1, originalMatrix = NULL, removeNumbers = FALSE, 
                           removePunctuation = TRUE, removeSparseTerms = 0, removeStopwords = TRUE, 
                           stemWords = FALSE, stripWhitespace = TRUE, toLower = TRUE, 
                           weighting = weightTf) 
{
  stem_words <- function(x) {
    split <- strsplit(x, " ")
    return(wordStem(unlist(split), language = language))
  }
  tokenize_ngrams <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = ngramLength))
  #tokenize_ngrams <- function(x, n = ngramLength) return(rownames(as.data.frame(unclass(textcnt(x, 
  #                                                                                              method = "string", n = n)))))
  BigramTokenizer <- function(x) NGramTokenizer(x, Weka_control(min = 1, max = ngramLength))
  
  
  control <- list(bounds = list(local = c(minDocFreq, maxDocFreq)), 
                  language = language, tolower = toLower, removeNumbers = removeNumbers, 
                  removePunctuation = removePunctuation, stopwords = removeStopwords, 
                  stripWhitespace = stripWhitespace, wordLengths = c(minWordLength, 
                                                                     maxWordLength), weighting = weighting)
  if (ngramLength > 1) {
    control <- append(control, list(tokenize = BigramTokenizer), 
                      after = 7)
  }
  else {
    control <- append(control, list(tokenize = scan_tokenizer), 
                      after = 4)
  }
  if (stemWords == TRUE && ngramLength == 1) 
    control <- append(control, list(stemming = stem_words), 
                      after = 7)
  trainingColumn <- apply(as.matrix(textColumns), 1, paste, 
                          collapse = " ")
  trainingColumn <- sapply(as.vector(trainingColumn, mode = "character"), 
                           iconv, to = "UTF8", sub = "byte")
  corpus <- VCorpus(VectorSource(trainingColumn), readerControl = list(language = language))
  matrix <- DocumentTermMatrix(corpus, control = control)
  if (removeSparseTerms > 0) 
    matrix <- removeSparseTerms(matrix, removeSparseTerms)
  if (!is.null(originalMatrix)) {
    terms <- colnames(originalMatrix[, which(!colnames(originalMatrix) %in% 
                                               colnames(matrix))])
    weight <- 0
    if (attr(weighting, "acronym") == "tf-idf") 
      weight <- 1e-09
    amat <- matrix(weight, nrow = nrow(matrix), ncol = length(terms))
    colnames(amat) <- terms
    rownames(amat) <- rownames(matrix)
    fixed <- as.DocumentTermMatrix(cbind(matrix[, which(colnames(matrix) %in% 
                                                          colnames(originalMatrix))], amat), weighting = weighting)
    matrix <- fixed
  }
  matrix <- matrix[, sort(colnames(matrix))]
  gc()
  return(matrix)
}

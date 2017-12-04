

filePath = '/Users/mjankowski/doc/workspace/data/reducedData.csv'

dataAll <- read.csv(filePath, sep=",")  
dim(dataAll)

labelMapping <- data.frame(label = c("ANDROID_TOOL", "KEYBOARD", "GAME", "NONE", "WIDGET", "USE_INTERNET", 
                                     "DOCUMENT_EDITOR", "LOCATE_POSITION", "APP_LIBRARY", "INTERNET_BROWSER", 
                                     "MESSAGING", "WALLPAPER", "WEATHER", "USE_CONTACTS", "BACKUP", "WORKOUT_TRACKING", 
                                     "CALENDAR", "MONEY", "GPS_NAVIGATION", "FLASHLIGHT", "HOME_LOCK_SCREEN", 
                                     "SMS", "JOB_SEARCH", "EBANKING", "CONTACT_MANAGER"), label = c(0:24))

dataAllLabelsNumeric <- labelsToNumeric(dataAll, labelMapping)

names(dataAllLabelsNumeric)[names(dataAllLabelsNumeric) == "label.1"] = "label"

#tfidfData <- prepareTfIdfWithLabelsTrainOnly(dataAll)

train_code <- as.factor(dataAll$label)

train_doc_matrix <- create_matrix(dataAll$text, language="english", removeNumbers=FALSE, stemWords=FALSE, 
                                  ngramLength=1, minWordLength = 1, removePunctuation = FALSE, removeStopwords = FALSE,
                                  stripWhitespace = FALSE, toLower = FALSE)

result <- FindTopicsNumber(
  train_doc_matrix,
  topics = c(10),
  metrics = c("Griffiths2004"),
  method = "Gibbs",
  control = list(seed = 77),
  mc.cores = 4L,
  verbose = TRUE
)

result$Griffiths2004

#rowTotals <- apply(train_doc_matrix, 1, sum) #Find the sum of words in each Document
#cleanedTrainMatrix   <- train_doc_matrix[rowTotals> 0, ]
#cleanedTrainLabels <- train_code[rowTotals> 0]

#dim(tfidfData$cleanedTrainMatrix)

topicmodel <- LDA(train_doc_matrix, k=10, control=list(seed=SEED))
trainData <- posterior(topicmodel)[2]$topics

trainData[3,]

res <- estimateTopicsCount4Methods(10,10,1, tfidfData = tfidfData)

res$ldatuningResults

prepareTfIdfWithLabelsTrainOnly <- function(data, sparseLevel=.998, ngramCount = 1){
  
  start.time <- Sys.time()
  train_code <- as.factor(data$label)
  
  train_doc_matrix <- create_matrix(data$text, language="english", removeNumbers=FALSE, stemWords=FALSE, 
                                    removeSparseTerms=sparseLevel, ngramLength=ngramCount)
  
  rowTotals <- apply(train_doc_matrix, 1, sum) #Find the sum of words in each Document
  cleanedTrainMatrix   <- train_doc_matrix[rowTotals> 0, ]
  cleanedTrainLabels <- train_code[rowTotals> 0]
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  tfIdfTrainMatrix=weightTfIdf(cleanedTrainMatrix)

  list(cleanedTrainMatrix=cleanedTrainMatrix, cleanedTrainLabels=cleanedTrainLabels,
       duration=duration)
}

estimateTopicsCount4MethodsRangeTrainOnly <- function(range, tfidfData, methods = c("Griffiths2004", "CaoJuan2009", "Arun2010", "Deveaud2014")){
  
  
  start.time <- Sys.time()
  
  result <- FindTopicsNumber(
    tfidfData,
    topics = range,
    metrics = methods,
    method = "Gibbs",
    control = list(seed = 77),
    mc.cores = 4L,
    verbose = TRUE
  )
  end.time <- Sys.time()
  time.taken <- end.time - start.time
  duration <- time.taken
  
  list(ldatuningResults=result, duration=duration)
}


sum(as.numeric(lapply(0:10, function(x){lgamma(x+0.1)}))) - lgamma(11*0.1)

lgamma(0)

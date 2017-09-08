
setwd('C:/doc/s/sem2/chudy/repo/pub_lda')
source('./preprocessing.R')
source('./dimRed.R')
source('./classification.R')
source('./validation.R')
source('./choose_model.R')
source('./plsa_em.R')

################################## PLSA ################################



topic_number <- 10

filePath = 'C:/doc/s/sem2/chudy/repo/pub_lda/apps_desc.csv'

data <- read.csv(filePath, sep=";")  
dim(data)

dataAll <- cleanData(data)

labelMapping <- data.frame(label = c("ANDROID_TOOL", "KEYBOARD", "GAME", "NONE", "WIDGET", "USE_INTERNET", 
                                     "DOCUMENT_EDITOR", "LOCATE_POSITION", "APP_LIBRARY", "INTERNET_BROWSER", 
                                     "MESSAGING", "WALLPAPER", "WEATHER", "USE_CONTACTS", "BACKUP", "WORKOUT_TRACKING", 
                                     "CALENDAR", "MONEY", "GPS_NAVIGATION", "FLASHLIGHT", "HOME_LOCK_SCREEN", 
                                     "SMS", "JOB_SEARCH", "EBANKING", "CONTACT_MANAGER"), label = c(0:24))

dataAllLabelsNumeric <- labelsToNumeric(dataAll, labelMapping)
names(dataAllLabelsNumeric)[names(dataAllLabelsNumeric) == "label.1"] = "label"

dim(dataAllLabelsNumeric)
partitioned <- partitionData(dataAllLabelsNumeric)
dim(partitioned$train)
dim(partitioned$test)

tfidfData <- prepareTfIdfWithLabels(partitioned, sparseLevel=0.98)

tf <- t(tfidfData$cleanedTrainMatrix)
tf
dim(tf)
M <- dim(tf)[2]
V <- dim(tf)[1]
M
V
K <- topic_number
K

m_tf <- as.matrix(tf)

liks <- plsaEM(K, m_tf, iter = 500)



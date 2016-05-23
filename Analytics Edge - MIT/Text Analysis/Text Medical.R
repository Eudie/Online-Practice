setwd("F:/Analytics Edge")

#loading data
  trials <- read.csv("clinical_trial.csv", stringsAsFactors = F)
  str(trials)
  max(nchar(trials$abstract))
  summary(trials)
  x <- nchar(trials$abstract)
  table(x)
  trials$title[nchar(trials$title)==min(nchar(trials$title))]
  
#Making Document term matrix
  library(tm)
  library(SnowballC)
  #For Title
    corpusTitle  <- Corpus(VectorSource(trials$title))
    corpusTitle  <- tm_map(corpusTitle , tolower)
    corpusTitle  <- tm_map(corpusTitle , PlainTextDocument)
    corpusTitle  <- tm_map(corpusTitle , removePunctuation)
    corpusTitle  <- tm_map(corpusTitle , removeWords, stopwords("english"))
    corpusTitle  <- tm_map(corpusTitle , stemDocument)
    dtmTitle     <- DocumentTermMatrix(corpusTitle)
    sparseTitle  <- removeSparseTerms(dtmTitle, 0.95)
    dtmTitle     <- as.data.frame(as.matrix(sparseTitle))
    
    #For Abstarct
    corpusAbstarct  <- Corpus(VectorSource(trials$abstract))
    corpusAbstarct  <- tm_map(corpusAbstarct , tolower)
    corpusAbstarct  <- tm_map(corpusAbstarct , PlainTextDocument)
    corpusAbstarct  <- tm_map(corpusAbstarct , removePunctuation)
    corpusAbstarct  <- tm_map(corpusAbstarct , removeWords, stopwords("english"))
    corpusAbstarct  <- tm_map(corpusAbstarct , stemDocument)
    dtmAbstarct     <- DocumentTermMatrix(corpusAbstarct)
    sparseAbstarct  <- removeSparseTerms(dtmAbstarct, 0.95)
    dtmAbstarct     <- as.data.frame(as.matrix(sparseAbstarct))
    
#Most frequent word in abstarct
  x <- colSums(dtmAbstarct)
  which.max(x)
  
#Combining both DTM
  colnames(dtmTitle) = paste0("T", colnames(dtmTitle))
  colnames(dtmAbstarct) = paste0("A", colnames(dtmAbstarct))
  dtm = cbind(dtmTitle, dtmAbstarct)
  dtm$trial <- trials$trial
  library(caTools)
  set.seed(144)
  split <- sample.split(dtm$trial, SplitRatio = 0.7)
  train <- dtm[split == T,]
  test <- dtm[split == F,]
  nrow(test[test$trial == 0,])/nrow(test)
  
#CART model
  library(rpart)
  library(rpart.plot)
  trialCART <- rpart(trial ~ ., data = train, method="class")
  prp(trialCART)
  pred <- predict(trialCART,  type="class")
  table(pred, train$trial)
  predTest <- predict(trialCART, newdata = test, type="class")
  Result <- as.matrix(table(predTest, test$trial))
  sum(Result[row(Result)== col(Result)])/sum(Result)
  
#AUC
  PredC <- predict(trialCART, newdata = test, type = "prob")
  PredROCRC <- prediction(PredC[,2], test$trial)
  as.numeric(performance(PredROCRC, "auc")@y.values)
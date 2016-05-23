setwd("F:/Analytics Edge")

#loading data
emails <- read.csv("emails.csv", stringsAsFactors = F)
table(emails$spam)
max(nchar(emails$text))
which.min(nchar(emails$text))

#Making Document term matrix
library(tm)
library(SnowballC)
corpusTitle  <- Corpus(VectorSource(emails$text))
corpusTitle  <- tm_map(corpusTitle , tolower)
corpusTitle  <- tm_map(corpusTitle , PlainTextDocument)
corpusTitle  <- tm_map(corpusTitle , removePunctuation)
corpusTitle  <- tm_map(corpusTitle , removeWords, stopwords("english"))
corpusTitle  <- tm_map(corpusTitle , stemDocument)
dtmTitle     <- DocumentTermMatrix(corpusTitle)
sparseTitle  <- removeSparseTerms(dtmTitle, 0.95)

emailsSparse <- as.data.frame(as.matrix(sparseTitle))
colnames(emailsSparse) <- make.names(colnames(emailsSparse))

x <- colSums(emailsSparse)
which.max(x)

emailsSparse$spam <- emails$spam
fiftho <- emailsSparse[emailsSparse$spam == 1, 1:330]
y <- colSums(fiftho)
y <- y[y>=1000]
length(y)

#machine leaning models
emailsSparse$spam = as.factor(emailsSparse$spam)
library(caTools)
set.seed(123)
split <- sample.split(emailsSparse$spam, SplitRatio = 0.7)
train <- emailsSparse[split == T,]
test <- emailsSparse[split == F,]

#Models
library(rpart)
library(rpart.plot)
library(randomForest)
spamLog <- glm(spam ~ ., data = train, family = "binomial")
spamCART <- rpart(spam ~ ., data = train, method =  "class")
spamRF <- randomForest(spam ~ ., data = train)

predLog <- predict(spamLog)
predCART <- predict(spamCART)
predRF <- predict(spamRF, type = "prob")

#Overfitting of log model
length(predLog[predLog<0.00001])
length(predLog[predLog>0.99999])
length(predLog) - (length(predLog[predLog<0.00001]) + length(predLog[predLog>0.99999]))

#Checking the models
library(ROCR)
#Log
summary(spamLog)
prp(spamCART)
table(predLog>0.5, train$spam)
PredROCRL <- prediction(predLog, train$spam)
as.numeric(performance(PredROCRL, "auc")@y.values)
#CART
Result <- as.matrix(table(predCART[,2] > 0.5, train$spam))
sum(Result[row(Result)== col(Result)])/sum(Result)
PredROCRC <- prediction(predCART[,2], train$spam)
as.numeric(performance(PredROCRC, "auc")@y.values)
#Randomforest
Result <- as.matrix(table(predRF[,2] > 0.5, train$spam))
sum(Result[row(Result)== col(Result)])/sum(Result)
PredROCRC <- prediction(predCART[,2], train$spam)
as.numeric(performance(PredROCRC, "auc")@y.values)

#Uptill now we are checking efficiency on training set. Now lets check on testing Set
predLog <- predict(spamLog, newdata = test)
predCART <- predict(spamCART, newdata = test)
predRF <- predict(spamRF, newdata = test, type = "prob")

#Checking the models
#Log
Result <- as.matrix(table(predLog > 0.5, test$spam))
sum(Result[row(Result)== col(Result)])/sum(Result)
PredROCRL <- prediction(predLog, test$spam)
as.numeric(performance(PredROCRL, "auc")@y.values)
#CART
Result <- as.matrix(table(predCART[,2] > 0.5, test$spam))
sum(Result[row(Result)== col(Result)])/sum(Result)
PredROCRC <- prediction(predCART[,2], test$spam)
as.numeric(performance(PredROCRC, "auc")@y.values)
#Randomforest
Result <- as.matrix(table(predRF[,2] > 0.5, test$spam))
sum(Result[row(Result)== col(Result)])/sum(Result)
PredROCRC <- prediction(predRF[,2], test$spam)
as.numeric(performance(PredROCRC, "auc")@y.values)


#The End
#Document classification, based on training set of 5486 observations, Can't submit because Hacker rank editor not supporting libraries.
#Solved for only one test case
#Possible Improvements: selecting optimum cv, random forest, XgBoost
#MyWorkingDirectory : setwd("F:/Hackerrank")

#libraries to be used
library(tm)
library(SnowballC)
library(rpart)
library(rpart.plot)


# Readind training data
train <- readLines("trainingdata.txt")
train <- train[2:5486]
train <- as.data.frame(cbind(substring(train, 1, 1), as.character(substring(train, 2))))
colnames(train) <- c("Type", "Document")

#Reading Test data
test <- readLines("Test.txt")
test <- as.data.frame(as.character(test[2:(6 + 1)]))
test <- cbind("a",test)
colnames(test) <- c("handel","Document")

#Combining Test and Train to create common DTM
Combine <- c(as.character(train[,2]),as.character(test[,2]))

#Preprocessing
Corp <- Corpus(VectorSource(Combine))
Corp <- tm_map(Corp, tolower)
Corp <- tm_map(Corp, PlainTextDocument)
Corp <- tm_map(Corp, removePunctuation)
Corp <- tm_map(Corp, removeWords, stopwords("english"))
Corp <- tm_map(Corp , stemDocument)
freq <- DocumentTermMatrix(Corp)
sparse <- removeSparseTerms(freq, 0.995)
corp.asdatafra <- as.data.frame(as.matrix(sparse))
colnames(corp.asdatafra) <- make.names(colnames(corp.asdatafra))

#Seprating Train  and Test
TrainClean <- corp.asdatafra[ 1:5485,]
TestClean <- corp.asdatafra[5486:5491,]

#Adding Factor variable to Train
TrainClean$Type <- train$Type


#Model CART
TreeModel <- rpart(Type ~ . , data = TrainClean,  method = "class")
prp(TreeModel)



#Prediction based on CART
stdout <- predict(TreeModel, newdata = TestClean, type = "class" )




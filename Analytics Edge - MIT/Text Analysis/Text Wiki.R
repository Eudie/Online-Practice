setwd("F:/Analytics Edge")

#Loading Data
  wiki <- read.csv("wiki.csv", stringsAsFactors = F)
  wiki$Vandal <- as.factor(wiki$Vandal)
  table(wiki$Vandal)

#Creating Documemt term matrix
  library(tm)
  library(SnowballC)
  Corp <- Corpus(VectorSource(wiki$Added))
  Corp <- tm_map(Corp, removeWords, stopwords("english"))
  Corp <- tm_map(Corp , stemDocument)
  dtmAdded <- DocumentTermMatrix(Corp)

#Saprse
  sparseAdded <- removeSparseTerms(dtmAdded, 0.997)
  wordsAdded <- as.data.frame(as.matrix(sparseAdded))
  colnames(wordsAdded) = paste("A", colnames(wordsAdded))
  
  #Replicating for removed words
    CorpR <- Corpus(VectorSource(wiki$Removed))
    CorpR <- tm_map(CorpR, removeWords, stopwords("english"))
    CorpR <- tm_map(CorpR , stemDocument)
    dtmRemoved <- DocumentTermMatrix(CorpR)
    
    
    
    
    #Defining Sparsity
      sparseRemoved <- removeSparseTerms(dtmRemoved, 0.997)
      wordsRemoved <- as.data.frame(as.matrix(sparseRemoved))
      colnames(wordsRemoved) = paste("R", colnames(wordsRemoved))
      
      
#Recreating cleaned data
  wikiWords <- cbind(wordsAdded, wordsRemoved)
  wikiWords$Vandal <- wiki$Vandal
  library(caTools)
  set.seed(123)
  split <- sample.split(wikiWords$Vandal, SplitRatio = 0.7)
  train <- wikiWords[split == T,]
  test <- wikiWords[split == F,]
  nrow(test[test$Vandal == 0,])/nrow(test)
  
#CART model
  library(rpart)
  library(rpart.plot)
  CartModel <- rpart(Vandal ~ ., data = train, method="class")
  Pred <- predict(CartModel, newdata = test, type = "class")
  Result <- as.matrix(table(Pred, test$Vandal))
  sum(Result[row(Result)== col(Result)])/sum(Result)
  prp(CartModel)
  
#Finding URLs
  wikiWords2 = wikiWords
  wikiWords2$HTTP = ifelse(grepl("http",wiki$Added,fixed=TRUE), 1, 0)
  table(wikiWords2$HTTP)
  
  # Making CArt model using new dataset
  wikiTrain2 = subset(wikiWords2, split==TRUE)
  wikiTest2 = subset(wikiWords2, split==FALSE)
  
  CartModel2 <- rpart(Vandal ~ ., data = wikiTrain2, method="class")
  Pred2 <- predict(CartModel2, newdata = wikiTest2, type = "class")
  Result2 <- as.matrix(table(Pred2, wikiTest2$Vandal))
  sum(Result2[row(Result2)== col(Result2)])/sum(Result2)
  
#FInding total number of words added or removed
  wikiWords2$NumWordsAdded = rowSums(as.matrix(dtmAdded))
  wikiWords2$NumWordsRemoved = rowSums(as.matrix(dtmRemoved))
  mean(wikiWords2$NumWordsAdded)
  
  #RECOMPUTE: Line 60-67
  
#Using Metadata to improve
  wikiWords3 = wikiWords2
  wikiWords3$Minor = wiki$Minor
  wikiWords3$Loggedin = wiki$Loggedin
  
  #Cart again
  wikiTrain3 = subset(wikiWords3, split==TRUE)
  wikiTest3 = subset(wikiWords3, split==FALSE)
  
  CartModel3 <- rpart(Vandal ~ ., data = wikiTrain3, method="class")
  Pred3 <- predict(CartModel3, newdata = wikiTest3, type = "class")
  Result3 <- as.matrix(table(Pred3, wikiTest3$Vandal))
  sum(Result3[row(Result3)== col(Result3)])/sum(Result3)
  prp(CartModel3)
  
#The End
  
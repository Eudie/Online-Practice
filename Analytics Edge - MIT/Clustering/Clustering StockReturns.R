setwd("F:/Analytics Edge")

#loading data
  stocks <- read.csv("StocksCluster.csv")
  summary(stocks)
  table(stocks[,12])
  cor(stocks)
  colMeans(stocks)

#Logistic Regression
  library(caTools)
  set.seed(144)
  spl = sample.split(stocks$PositiveDec, SplitRatio = 0.7)
  stocksTrain = subset(stocks, spl == TRUE)
  stocksTest = subset(stocks, spl == FALSE)
  
  StocksModel <- glm(PositiveDec ~ ., data = stocksTrain, family = 'binomial')
  pred <- predict(StocksModel, newdata = stocksTest, type = 'response')
  ResultC <- as.matrix(table(pred >0.5, stocksTest$PositiveDec))
  sum(ResultC[row(ResultC)== col(ResultC)])/sum(ResultC)

#Clustering
  #Removing Depennt variable
    limitedTrain = stocksTrain
    limitedTrain$PositiveDec = NULL
    limitedTest = stocksTest
    limitedTest$PositiveDec = NULL
  
  #Normalizing
    library(caret)
    preproc = preProcess(limitedTrain)
    normTrain = predict(preproc, limitedTrain)
    normTest = predict(preproc, limitedTest)
  
  #Kmean method
    set.seed(144)
    km <- kmeans(normTrain, 3)
    
    library(flexclust)
    km.kcca = as.kcca(km, normTrain)
    clusterTrain = predict(km.kcca)
    clusterTest = predict(km.kcca, newdata=normTest)
  
  #assigning clusters
    stocksTrain1 <- stocksTrain[clusterTrain == 1,]
    stocksTrain2 <- stocksTrain[clusterTrain == 2,]
    stocksTrain3 <- stocksTrain[clusterTrain == 3,]
    stocksTest1   <- stocksTest[clusterTest == 1,]
    stocksTest2   <- stocksTest[clusterTest == 2,]
    stocksTest3   <- stocksTest[clusterTest == 3,]
  
  #Logisting model by clusters
    StocksModel1 <- glm(PositiveDec ~ ., data = stocksTrain1, family = 'binomial')
    StocksModel2 <- glm(PositiveDec ~ ., data = stocksTrain2, family = 'binomial')
    StocksModel3 <- glm(PositiveDec ~ ., data = stocksTrain3, family = 'binomial')
    summary(StocksModel1)
    summary(StocksModel2)
    summary(StocksModel3)
  
  #Predictions
    PredictTest1 <- predict(StocksModel1, newdata = stocksTest1, type = 'response')
    ResultC <- as.matrix(table(PredictTest1 >0.5, stocksTest1$PositiveDec))
    sum(ResultC[row(ResultC)== col(ResultC)])/sum(ResultC)
    
    PredictTest2 <- predict(StocksModel2, newdata = stocksTest2, type = 'response')
    ResultC <- as.matrix(table(PredictTest2 >0.5, stocksTest2$PositiveDec))
    sum(ResultC[row(ResultC)== col(ResultC)])/sum(ResultC)
    
    PredictTest3 <- predict(StocksModel3, newdata = stocksTest3, type = 'response')
    ResultC <- as.matrix(table(PredictTest3 >0.5, stocksTest3$PositiveDec))
    sum(ResultC[row(ResultC)== col(ResultC)])/sum(ResultC)
    
      #overall
        AllPredictions = c(PredictTest1, PredictTest2, PredictTest3)
        AllOutcomes = c(stocksTest1$PositiveDec, stocksTest2$PositiveDec, stocksTest3$PositiveDec)
        ResultC <- as.matrix(table(AllPredictions >0.5, AllOutcomes))
        sum(ResultC[row(ResultC)== col(ResultC)])/sum(ResultC)

#The End
setwd("F:/Analytics Edge")

library(rpart)
library(rpart.plot)
library(ROCR)
library(caTools)
library(randomForest)

#Reading and splitting Data
  census <- read.csv("census.csv")
  set.seed(2000)
  split <- sample.split(census$over50k, SplitRatio = 0.6)
  train <- census[split == T,]
  test <- census[split == F,]

#Logistic Model
  LogModel <- glm(over50k ~ ., data = train, family = "binomial")
  summary(LogModel)
  PredL <- predict(LogModel, newdata = test, type = "response")
  table(PredL>0.5, test$over50k)
  PredROCR <- prediction(PredL, test$over50k)
  as.numeric(performance(PredROCR, "auc")@y.values)

#CART model
  CartModel <- rpart(over50k ~ ., method="class", data = train)
  prp(CartModel)
  PredC <- predict(CartModel, newdata = test, type = "prob")
  ResultC <- as.matrix(table(PredC > 0.5, test$over50k))
  sum(ResultC[row(ResultC)== col(ResultC)])/sum(ResultC)
  PredROCRC <- prediction(PredC[,2], test$over50k)
  as.numeric(performance(PredROCRC, "auc")@y.values)

#Random Forest
  set.seed(1)
  trainSmall = train[sample(nrow(train), 2000),]
  ForestModel <- randomForest(over50k ~ ., data = trainSmall, method = "class")
  PredF <- predict(ForestModel, newdata = test)
  ResultF <- as.matrix(table(PredF , test$over50k))
  sum(ResultF[row(ResultF)== col(ResultF)])/sum(ResultF)
  
  #Important variable in Random Forest
    vu = varUsed(ForestModel, count=TRUE)
    vusorted = sort(vu, decreasing = FALSE, index.return = TRUE)
    dotchart(vusorted$x, names(ForestModel$forest$xlevels[vusorted$ix]))
  #Impurity
    varImpPlot(ForestModel)

#Cross validation for CART Model
  library(caret)
  library(e1071)
  set.seed(2)
  fold <- trainControl(method = "cv", number = 10)
  cartGrid = expand.grid( .cp = seq(0.002,0.1,0.002))
  train(over50k ~ ., data = train, method = "rpart", trControl = fold, tuneGrid = cartGrid)
  
  #CART by suggested cp
    CartModelcp <- rpart(over50k ~ ., method="class", data = train, cp = 0.002)
    PredCcp <- predict(CartModelcp, newdata = test, type = "class")
    ResultCcp <- as.matrix(table(PredCcp, test$over50k))
    sum(ResultCcp[row(ResultCcp)== col(ResultCcp)])/sum(ResultCcp)
    prp(CartModelcp)

#The End



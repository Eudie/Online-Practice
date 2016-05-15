setwd("F:/Analytics Edge")

library(rpart)
library(rpart.plot)
library(ROCR)
library(caTools)
library(randomForest)

#For single factor value
  #Data load
    letters <- read.csv("letters_ABPR.csv")
    letters$isB = as.factor(letters$letter == "B")

  #Spliting data in training and testing Set
    set.seed(1000)
    split <- sample.split(letters$isB, SplitRatio = 0.5)
    train <- letters[split == TRUE,]
    test <- letters[split == FALSE,]
    table(test$isB)
                          
  #CART model
    CARTb = rpart(isB ~ . - letter, data=train, method="class")
    pred <- predict(CARTb, newdata = test, type = "class")
    table(pred, test$isB)
  
  #Random Forest model
    Forest <- randomForest(isB ~ . -letter, data = train, method = "class")
    Rpred <- predict(Forest, newdata = test, type = "class")
    table(Rpred, test$isB)

#For multiple factor values
    letters$letter = as.factor( letters$letter )
    set.seed(2000)
    SplitM <- sample.split(letters$letter, SplitRatio = 0.5)
    TrainM <- letters[SplitM == T,]
    TestM <- letters[SplitM == F,]
    
  #for baseline
    BaseV = as.vector(table(TestM$letter))
    BaselinePer <- max(BaseV)/sum(BaseV)
    BaselinePer
  
  #Cart Model
    CartM <- rpart(letter ~ . - isB, data = TrainM, method = "class")
    predM <- predict(CartM, newdata = TestM, type = "class")
    Result <- as.matrix(table(predM, TestM$letter))
    perform = sum(Result[row(Result)==col(Result)])/sum(Result)
    perform
  
  #Random Forest
    ForestM <- randomForest(letter ~ . - isB, data = TrainM, method = "class")
    FpredM <- predict(ForestM, newdata = TestM, type = "class")
    Result <- as.matrix(table(FpredM, TestM$letter))
    Fperform = sum(Result[row(Result)==col(Result)])/sum(Result)
    Fperform

#The End



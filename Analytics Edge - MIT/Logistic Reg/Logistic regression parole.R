setwd("F:/Analytics Edge")

parole <- read.csv("parole.csv")
table(parole$violator)
table(parole$male)
table(parole$race)
table(parole$state)
table(parole$multiple.offenses)
table(parole$crime)
parole$state <- as.factor(parole$state)
parole$crime <- as.factor(parole$crime)
summary(parole)

library(caTools)
set.seed(144)
split = sample.split(parole$violator, SplitRatio = 0.7)
train = subset(parole, split == TRUE)
test = subset(parole, split == FALSE)

Model0 <- glm(violator ~ ., data = train, family = "binomial")
summary(Model0)
criminal <- c(1,1,50,1,3,12,0,2,0)
test2 <- rbind(test,criminal)
test3 <- test2[203,]


Predicttion <- predict(Model0, type = "response", newdata = test3)
Predicttion

PredTest <- predict(Model0, type = "response", newdata = test)
max(PredTest)

table(PredTest >0.5, test$violator)

# Test set AUC 
library(ROCR)
ROCRpred = prediction(PredTest, test$violator)
as.numeric(performance(ROCRpred, "auc")@y.values)
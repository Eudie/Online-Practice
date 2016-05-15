setwd("F:/Analytics Edge")

gerber <- read.csv("gerber.csv")
LogMod <- glm(voting ~ civicduty + hawthorne + self + neighbors, data = gerber, family = "binomial")
pred <- predict(LogMod, type = "response")
library(ROCR)
rocrpre <- prediction(pred, gerber$voting)
as.numeric(performance(rocrpre, "auc")@y.values)

#Cart model
library(rpart)
library(rpart.plot)
CARTmodel = rpart(voting ~ civicduty + hawthorne + self + neighbors, data=gerber)
plot(CARTmodel)
CARTmodel2 = rpart(voting ~ civicduty + hawthorne + self + neighbors, data=gerber, cp=0.0)
prp(CARTmodel2)
CARTmodel3 = rpart(voting ~ sex + civicduty + hawthorne + self + neighbors, data=gerber, cp=0.0)
prp(CARTmodel3)
CARTmodel4 = rpart(voting ~ control , data=gerber, cp=0.0)
prp(CARTmodel4, digits = 6)
CARTmodel5 = rpart(voting ~ control + sex , data=gerber, cp=0.0)
prp(CARTmodel5, digits = 6)

#check by logistic model
LogModelSex <- glm(voting ~ control + sex, data = gerber, family = "binomial")
summary(LogModelSex)
Possibilities = data.frame(sex=c(0,0,1,1),control=c(0,1,0,1))
predict(LogModelSex, newdata=Possibilities, type="response")

#LogModel for combination of ssex and control
LogModel2 = glm(voting ~ sex + control + sex:control, data=gerber, family="binomial")
summary(LogModel2)
predict(LogModel2, newdata=Possibilities, type="response")

#The End

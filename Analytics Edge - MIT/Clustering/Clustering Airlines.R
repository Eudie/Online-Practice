setwd("F:/Analytics Edge")

#loading data
  airlines <- read.csv("AirlinesCluster.csv")
  summary(airlines)

#Normalising the data
  library(caret)
  preproc <- preProcess(airlines)
  airlinesNorm <- predict(preproc, airlines)
  summary(airlinesNorm)

#Hierarical Clustering
  distance <- dist(airlinesNorm, method = "euclidean")
  cluster <- hclust(distance, method = "ward.D")
  plot(cluster)
  clustergroup <- cutree(cluster, k = 5)
  
  #Summary of Clusters
    tapply(airlines$Balance, clustergroup, mean)
    tapply(airlines$QualMiles, clustergroup, mean)
    tapply(airlines$BonusMiles, clustergroup, mean)
    tapply(airlines$BonusTrans, clustergroup, mean)
    tapply(airlines$FlightMiles, clustergroup, mean)
    tapply(airlines$FlightTrans, clustergroup, mean)
    tapply(airlines$DaysSinceEnroll, clustergroup, mean)

#K-mean clustering
  set.seed(88)
  KCluster <- kmeans(airlinesNorm,iter.max = 1000 ,5)
  table(KCluster[1])

#The End
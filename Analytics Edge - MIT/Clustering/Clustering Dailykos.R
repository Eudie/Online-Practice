setwd("F:/Analytics Edge")

#loading data
dailykos <- read.csv("dailykos.csv")
str(dailykos)

#Hierarical Clustering
  distance <- dist(dailykos, method = "euclidean")
  cluster <- hclust(distance, method = "ward.D")
  plot(cluster)
  clustergroup <- cutree(cluster, k = 7)
  
  #Dividing dataset into clusters
    C1 <- dailykos[clustergroup == 1,]
    C2 <- dailykos[clustergroup == 2,]
    C3 <- dailykos[clustergroup == 3,]
    C4 <- dailykos[clustergroup == 4,]
    C5 <- dailykos[clustergroup == 5,]
    C6 <- dailykos[clustergroup == 6,]
    C7 <- dailykos[clustergroup == 7,]
  
  #most frequent word
    tail(sort(colMeans(C1)))
    tail(sort(colMeans(C2)))
    tail(sort(colMeans(C3)))
    tail(sort(colMeans(C4)))
    tail(sort(colMeans(C5)))
    tail(sort(colMeans(C6)))
    tail(sort(colMeans(C7)))

#K-mean clustering
  set.seed(1000)
  KCluster <- kmeans(dailykos, 7)
  table(KCluster[1])
  
  
  #Dividing dataset into clusters
    ClusterNo <- as.data.frame(KCluster[1])
    kC1 <- dailykos[ClusterNo == 1,]
    kC2 <- dailykos[ClusterNo == 2,]
    kC3 <- dailykos[ClusterNo == 3,]
    kC4 <- dailykos[ClusterNo == 4,]
    kC5 <- dailykos[ClusterNo == 5,]
    kC6 <- dailykos[ClusterNo == 6,]
    kC7 <- dailykos[ClusterNo == 7,]
    
  #most frequent word
    tail(sort(colMeans(kC1)))
    tail(sort(colMeans(kC2)))
    tail(sort(colMeans(kC3)))
    tail(sort(colMeans(kC4)))
    tail(sort(colMeans(kC5)))
    tail(sort(colMeans(kC6)))
    tail(sort(colMeans(kC7)))

#The End
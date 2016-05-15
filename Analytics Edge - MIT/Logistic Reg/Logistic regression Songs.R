setwd("F:/Analytics Edge")

songs <- read.csv("songs.csv")
table(songs$year)
table(songs$artistname)
Micheal <- songs[songs$artistname == "Michael Jackson" & songs$Top10 == 1,]
Micheal$songtitle

table(songs$timesignature)
HigestTempo <- songs$songtitle[songs$tempo == max(songs$tempo)]
HigestTempo

SongsTrain = subset(songs, year <= 2009)
SongsTest = subset(songs, year == 2010)


nonvars = c("year", "songtitle", "artistname", "songID", "artistID")
SongsTrain = SongsTrain[ , !(names(SongsTrain) %in% nonvars) ]
SongsTest = SongsTest[ , !(names(SongsTest) %in% nonvars) ]
SongsLog1 = glm(Top10 ~ ., data=SongsTrain, family=binomial)
SongsLog1

cor(SongsTrain$loudness, SongsTrain$energy)

SongsLog2 = glm(Top10 ~ . - loudness, data=SongsTrain, family=binomial)
SongsLog2

SongsLog3 = glm(Top10 ~ . - energy, data=SongsTrain, family=binomial)
SongsLog3

PredictTest <- predict(SongsLog3, type = "response", newdata = SongsTest)
table( PredictTest > 0.45,SongsTest$Top10)

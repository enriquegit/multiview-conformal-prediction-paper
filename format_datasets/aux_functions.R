library(foreign)
library(R.matlab)
library(data.table)
library(tuneR)

normalize <- function(x){
  #x is a data frame
  for(i in 1:ncol(x)){
    c <- x[,i]
    max <- max(c)
    min <- min(c)
    if(max==min){
      x[,i] <- max
    }
    else{
      x[,i] <- (c - min) / (max - min)
    }
  }
  x
}

movingAverage <- function(x, n){
  #applies moving average filter to x with window of size n
  N <- length(x); s <- numeric(length=N)
  for(i in n:N){s[i] <- sum(x[(i-n+1):i]) / n}
  for(i in 1:(n-1)){s[i] <- sum(x[i:(i+n-1)]) / n}
  return(s)
}

f.meanDif <- function(x){
  acum <- 0.0
  for(i in 2:length(x)){
    acum <- acum + (x[i] - x[i-1])
  }
  mean(acum)
}

berkeley.processFile <- function(path, prefix){
  #prefix="s01_a01_r01"
  
  label <- strsplit(prefix, "_")[[1]][2]
  winLength <- 1 #window length in seconds
  samplingRate <- 48000
  samplePoints <- winLength * samplingRate
  
  instance <- data.frame(userid=1, label=label)
  
  #Mic01
  f <- paste0(path,"Audio/Mic01/aud_m01_",prefix,".wav")
  f.mic01 <- extractMicFeatures(f, winLength, samplingRate)
  names(f.mic01) <- paste0("v1_Mic01mfcc",1:length(f.mic01))
  instance <- cbind(instance,as.data.frame(t(f.mic01)))
  
  #Mic02
  f <- paste0(path,"Audio/Mic02/aud_m02_",prefix,".wav")
  f.mic02 <- extractMicFeatures(f, winLength, samplingRate)
  names(f.mic02) <- paste0("v1_Mic02mfcc",1:length(f.mic02))
  instance <- cbind(instance,as.data.frame(t(f.mic02)))
  
  #Mic03
  f <- paste0(path,"Audio/Mic03/aud_m03_",prefix,".wav")
  f.mic03 <- extractMicFeatures(f, winLength, samplingRate)
  names(f.mic03) <- paste0("v1_Mic03mfcc",1:length(f.mic03))
  instance <- cbind(instance,as.data.frame(t(f.mic03)))
  
  #Mic04
  f <- paste0(path,"Audio/Mic04/aud_m04_",prefix,".wav")
  f.mic04 <- extractMicFeatures(f, winLength, samplingRate)
  names(f.mic04) <- paste0("v1_Mic04mfcc",1:length(f.mic04))
  instance <- cbind(instance,as.data.frame(t(f.mic04)))
  
  #Accelerometer02
  f <- paste0(path,"Accelerometer/Shimmer02/acc_h02_",prefix,".txt")
  tmp <- extractAccFeatures(f)
  names(tmp) <- paste0("v2_Acc02",names(tmp))
  instance <- cbind(instance, tmp)
  
  
  #Skeleton
  f <- paste0(path,"Mocap/SkeletalData/skl_",prefix,".bvh")
  skl <- read.table(f, header = F, quote = "", sep = " ", skip = 174)
  refPoint <- c(1,2,3)
  
  distances <- matrix(NA, nrow = 30, ncol = nrow(skl), dimnames = list(x=1:30,y=1:nrow(skl)))
  
  for(i in 1:nrow(skl)){
    frame <- skl[i,]
    joints <- seq(4, ncol(skl) - 1 , 3)
    #iterate joints
    for(idx in 1:length(joints)){
      j <- joints[idx]
      d <- dist(rbind(as.matrix(skl[i,refPoint]), as.matrix(skl[i,c(j,j+1,j+2)])), method = "manhattan")[[1]]
      distances[idx,i] <- d
    }
  }
  
  #distances
  means <- apply(distances, 1, mean); names(means) <- paste0("v3_meanDist",names(means))
  maxs <- apply(distances, 1, max); names(maxs) <- paste0("v3_maxDist",names(maxs))
  mins <- apply(distances, 1, min); names(mins) <- paste0("v3_minDist",names(mins))
  
  instance <- cbind(instance, matrix(means, ncol = length(means), dimnames = list(x=1,y=names(means))))
  instance <- cbind(instance, matrix(maxs, ncol = length(maxs), dimnames = list(x=1,y=names(maxs))))
  instance <- cbind(instance, matrix(mins, ncol = length(mins), dimnames = list(x=1,y=names(mins))))
  
  return(instance)
}

extractAccFeatures <- function(f){
  
  df <- read.table(f, sep = " ", header = F, quote = "")
  x <- 1; y <- 2; z <- 3;
  
  #moving average
  df[,x] <- movingAverage(df[,x], 10)
  df[,y] <- movingAverage(df[,y], 10)
  df[,z] <- movingAverage(df[,z], 10)
  
  meanX <- mean(df[,x]); meanY <- mean(df[,y]); meanZ <- mean(df[,z])
  sdX <- sd(df[,x]); sdY <- sd(df[,y]); sdZ <- sd(df[,z])
  maxX <- max(df[,x]); maxY <- max(df[,y]); maxZ <- max(df[,z])
  corXY <- cor(df[,x],df[,y]); corXZ <- cor(df[,x],df[,z]); corYZ <- cor(df[,y],df[,z])
  
  magnitude <- sqrt(df[,x]^2 + df[,y]^2 + df[,z]^2)
  
  meanMagnitude <- mean(magnitude)
  sdMagnitude <- sd(magnitude)
  auc <- sum(magnitude)
  meanDif <- f.meanDif(magnitude)
  
  tmp <- data.frame(meanX=meanX, meanY=meanY, meanZ=meanZ, sdX=sdX, sdY=sdY, sdZ=sdZ, maxX=maxX, maxY=maxY, maxZ=maxZ, corXY=corXY, corXZ=corXZ, corYZ=corYZ, meanMagnitude=meanMagnitude, sdMagnitude=sdMagnitude, auc=auc, meanDif=meanDif)
  
  return(tmp)
}

extractMicFeatures <- function(f, winLength, samplingRate){
  
  waveobj <- readWave(f)
  
  duration <- length(waveobj@left) / samplingRate
  if(duration < winLength){
    warning(paste0("audio file less than 1 sec.: ",f))
  }
  numSegments <- floor(duration / winLength)
  
  if(numSegments > 1){
    coeffs <- melfcc(waveobj, wintime = .9, hoptime = .9999)
  }
  else{
    coeffs <- melfcc(waveobj, wintime = 1, hoptime = .5)
  }
  f.mic <- colMeans(coeffs)
  
}

berkeley.processAll <- function(path, normalize=""){
  
  dataset <- NULL
  prefixes <- list.files(paste0(path,"Accelerometer/Shimmer02/"), pattern = "*.txt")
  splits <- strsplit(prefixes,"_")
  prefixes <- lapply(splits, function(e){
    paste(e[[3]],e[[4]],strsplit(e[[5]],"\\.")[[1]][[1]],sep = "_")
  })
  
  prefixes <- sapply(prefixes, c)
  
  for(i in 1:length(prefixes)){
    p <- prefixes[i]
    print(paste0("processing prefix = ",p))
    
    #check that p exists for all used sensors
    
    tmp <- berkeley.processFile(path, p)
    dataset <- rbind(dataset,tmp)
  }
  
  dataset <- dataset[complete.cases(dataset),]
  
  if(normalize=="by_all"){
    dataset <- cbind(dataset[,c(1,2)], normalize(dataset[,3:ncol(dataset)]))
  }
  
  return(dataset)
}

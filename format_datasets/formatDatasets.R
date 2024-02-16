source("aux_functions.R")


#### berkeley dataset ####
rawPath <- "C://bigdata//berkeley-mhad//"
dataset <- berkeley.processAll(rawPath, "by_all")
dataset <- dataset[,-1] # Remove id column
names(dataset)[1] <- "class"
write.csv(dataset, paste0(rawPath,"data.csv"), row.names = F, quote = F)

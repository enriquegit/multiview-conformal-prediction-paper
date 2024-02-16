source("format_results_functions.R")
library(reticulate)

reticulate::source_python("..//notebooks//globals.py")

pwidth <- 5; pheight <- 5 # Set the plots width and height.

#add.intersection.model(DATASET_PATH) # Use this for 3 views (berkeley)
#add.intersection.model.2v(DATASET_PATH) # Use this for 2 views (htad)

change.method.names(DATASET_PATH, DATASET_NAME)

res1 <- summarize.iterations(DATASET_PATH, "1")

res2 <- summarize.all(DATASET_PATH, "1")

pairwise.occurrences(DATASET_PATH, "1", pwidth, pheight)

plot.confusion.matrix(DATASET_PATH, "1", pwidth, pheight)

latex.summary(DATASET_PATH, "1")

plot.scatter(DATASET_PATH)

plot.histograms(DATASET_PATH, DATASET_NAME)

#### Box plots ####

df <- read.csv(paste0(DATASET_PATH,"results_1//summary_iterations.csv"))

pdf(paste0(DATASET_PATH,"results_1//boxplot_setsize.pdf"), 8, 7)
boxplot(setsize~method,df)
dev.off()

pdf(paste0(DATASET_PATH,"results_1//boxplot_F1.pdf"), 8, 7)
boxplot(F1~method,df)
dev.off()


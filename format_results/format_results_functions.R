library(caret)
library(ggplot2)
library(kableExtra)
library(ggpubr)
library(ggrepel)

coverage <- function(gt, ps){
  # Computes the conformal coverage.
  # gt: ground truth.
  # ps: prediction set.
  
  ps <- strsplit(ps, "\\|")
  
  n <- length(gt)
  
  hits <- 0
  
  for(i in 1:n){
    p <- gt[i]
    s <- ps[i][[1]]
    
    if(p %in% s){
      hits <- hits + 1
    }
  }
  
  return(hits / n)
}

avg.set.size <- function(ps){
  # Compute the average set size.
  # ps: prediction set.
  
  n <- length(ps)
  
  ps <- strsplit(ps, "\\|")
  
  sizes <- sapply(ps, function(e){
    length(e)
  })
  
  return(mean(sizes))
}

pct.empty <- function(ps){
  # Computes the percent of empty sets.
  n <- length(ps)
  
  ps <- strsplit(ps, "\\|")
  
  sizes <- sapply(ps, function(e){
    length(e)
  })
  
  pct <- sum(sizes == 0) / n
  
  return(pct)
}

observed.unconfidence <- function(gt, ps, pvalues, labels.order){
  # Smaller are preferred.
  # Omits empty sets.
  
  n <- length(gt)
  
  acum <- 0
  
  count <- 0
  
  for(i in 1:n){
    parts <- strsplit(ps[i], "\\|")[[1]]
    
    if(length(parts) == 0){
      next
    }
    else{
      count <- count + 1
      
      false.labels <- parts[!parts %in% gt[i]]
      
      if(length(false.labels) == 0){
        next
      }
      else{
        pvs <- as.numeric(strsplit(pvalues[i], "\\|")[[1]])
        
        acum <- acum + max(pvs[labels.order %in% false.labels])
      }
    }
  }
  
  return(acum/count)
}

observed.excess <- function(gt, ps){
  # OE.
  # Smaller is better.
  # Counts empty set as false label.
  
  n <- length(gt)
  
  acum <- 0
  
  for(i in 1:n){
    
    parts <- strsplit(ps[i], "\\|")[[1]]
    
    false.labels.count <- sum(!parts %in% gt[i])
    
    acum <- acum + false.labels.count
  }
  
  return(acum/n)
}

jaccard <- function(gt, ps){
  # Higher values are better.
  
  n <- length(gt)
  
  acum <- 0
  
  for(i in 1:n){
    
    parts <- strsplit(ps[i], "\\|")[[1]]
    
    if(length(parts) == 1 && parts == ""){
      acum <- acum + 0
    }
    else{
      tmp.intersection <- sum(parts %in% gt[i])
      tmp.union <- length(unique(c(parts,gt[i])))
      acum <- acum + (tmp.intersection / tmp.union)
    }
    
  }
  
  return(acum/n)
}

average.fuzziness <- function(pvalues){
  # F criterion.
  # Smaller values are preferred.
  
  n <- length(pvalues)
  
  acum <- 0
  
  for(i in 1:n){
    
    pvs <- as.numeric(strsplit(pvalues[i], "\\|")[[1]])
    
    acum <- acum + sum(sort(pvs, decreasing = T)[-1])
  }
  
  return(acum/n)
}

pct.m.criterion <- function(ps){
  # M criterion.
  # Smaller values are preferred.
  # Discards empty sets.
  
  n <- length(ps)
  
  acum <- 0
  
  count <- 0
  
  for(i in 1:n){
    
    parts <- strsplit(ps[i], "\\|")[[1]]
    
    if(length(parts) == 0){
      next
    }
    else{
      count <- count + 1
      if(length(parts) > 1){
        acum <- acum + 1
      }
    }
    
  }
  
  return(acum/count)
  
}

om.criterion <- function(gt, ps){
  # OM.
  # Small values are preferred.
  # Discard empty sets.
  
  n <- length(gt)
  
  acum <- 0
  
  count <- 0
  
  for(i in 1:n){
    
    parts <- strsplit(ps[i], "\\|")[[1]]
    
    if(length(parts) == 0){
      next
    }
    else{
      
      count <- count + 1
      
      false.labels.count <- sum(!parts %in% gt[i])
      
      if(false.labels.count > 1)false.labels.count <- 1;
      
      acum <- acum + false.labels.count
    }
  }
  
  return(acum/count)
}

observed.fuzziness <- function(gt, ps, pvalues, labels.order){
  # OF
  # Smaller are preferred.
  # Omits empty sets.
  
  n <- length(gt)
  
  acum <- 0
  
  count <- 0
  
  for(i in 1:n){
    
    parts <- strsplit(ps[i], "\\|")[[1]]
    
    if(length(parts) == 0){
      next
    }
    else{
      count <- count + 1
      
      false.labels <- parts[!parts %in% gt[i]]
      
      if(length(false.labels) == 0){
        next
      }
      else{
        pvs <- as.numeric(strsplit(pvalues[i], "\\|")[[1]])
        
        acum <- acum + sum(pvs[labels.order %in% false.labels])
      }
    }
    
  }
  
  return(acum/count)
}

change.method.names <- function(dataset_path, dataset_name="htad"){
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_1//results_mvc.csv"))
  
  mvc <- "MV-I"
  mvcs <- "MV-S"
  aggregated <- "MV-A"
  
  if(dataset_name == "htad"){
    v1 <- "audio"
    v2 <- "accelerometer"
  }
  else if(dataset_name == "berkeley"){
    v1 <- "audio"
    v2 <- "accelerometer"
    v3 <- "skeleton"
    
    results$method[results$method == "v3"] <- v3
  }
  
  results$method[results$method == "v1"] <- v1
  results$method[results$method == "v2"] <- v2
  results$method[results$method == "aggregated"] <- aggregated
  results$method[results$method == "mvcs"] <- mvcs
  results$method[results$method == "mvc"] <- mvc
  
  # Write results.
  write.csv(results, paste0(dataset_path,"results_1//results_mvc.csv"), row.names = F, quote = F)

}

add.intersection.model <- function(dataset_path){
  # Function that adds the predictions for the intersection model.
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_1//results.csv"))
  
  # Data frame that will contain the original plus the new model results.
  all <- results
  
  labels.order <- read.csv(paste0(dataset_path,"classes.csv"), header = F)[,1]
  
  df.v1 <- results[results$method == "v1",]
  df.v2 <- results[results$method == "v2",]
  df.v3 <- results[results$method == "v3",]
  
  for(i in 1:nrow(df.v1)){
    
    row.v1 <- df.v1[i,]
    row.v2 <- df.v2[i,]
    row.v3 <- df.v3[i,]
    
    # Create arrays to store sum of scores and counts.
    sum.scores <- rep(0, length(labels.order))
    names(sum.scores) <- labels.order
    counts <- sum.scores
    
    ps1 <- strsplit(row.v1$predictionSet, "\\|")[[1]]
    if(length(ps1) == 0){
      ps1 <- ""
    }
    else{
      sum.scores[ps1] <- as.numeric(strsplit(row.v1$scores, "\\|")[[1]])
      counts[ps1] <- counts[ps1] + 1 
    }
    
    ps2 <- strsplit(row.v2$predictionSet, "\\|")[[1]]
    if(length(ps2) == 0){
      ps2 <- ""
    }
    else{
      sum.scores[ps2] <- sum.scores[ps2] + as.numeric(strsplit(row.v2$scores, "\\|")[[1]])
      counts[ps2] <- counts[ps2] + 1
    }
    
    ps3 <- strsplit(row.v3$predictionSet, "\\|")[[1]]
    if(length(ps3) == 0){
      ps3 <- ""
    }
    else{
      sum.scores[ps3] <- sum.scores[ps3] + as.numeric(strsplit(row.v3$scores, "\\|")[[1]])
      counts[ps3] <- counts[ps3] + 1
    }
    
    # Compute average scores.
    avg.scores <- sum.scores / counts
    
    # Compute the intersection of the prediction sets.
    psInt <- intersect(intersect(ps1, ps2), ps3)
    
    
    if(length(psInt) > 0){
      tmp <- avg.scores[psInt]
      scores <- paste(tmp, collapse = "|")
    }
    else{
      scores <- ""
    }
    
    # Compute average p-values.
    pvs1 <- as.numeric(strsplit(row.v1$pvalues, "\\|")[[1]])
    pvs2 <- as.numeric(strsplit(row.v2$pvalues, "\\|")[[1]])
    pvs3 <- as.numeric(strsplit(row.v3$pvalues, "\\|")[[1]])
    
    avg.pvalues <- (pvs1 + pvs2 + pvs3) / 3
    
    avg.pvalues <- paste(avg.pvalues, collapse = "|")
    
    # Compute prediction.
    
    # If psInt != "" predict the one with highest score.
    # else take the max score from the union.
    if(length(psInt) > 0){
      prediction <- names(which.max(tmp))
    }
    else{
      prediction <- names(which.max(avg.scores))
    }
    
    if(length(psInt) == 0){
      #psInt <- ""
      psInt <- prediction
    }
    else{
      psInt <- paste(psInt, collapse = "|")
    }
    
    # Build data frame row and add it.
    df <- data.frame(it=row.v1$it, method="mvc",
                     groundTruth=row.v1$groundTruth,
                     prediction=prediction,
                     predictionSet=psInt,
                     scores=scores,
                     pvalues=avg.pvalues)
    
    all <- rbind(all, df)
    
  } # end for
  
  write.csv(all, paste0(dataset_path,"results_1//results_mvc.csv"), row.names = F, quote = F)
  
}

add.intersection.model.2v <- function(dataset_path){
  # Function that adds the predictions for the intersection model.
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_1//results.csv"))
  
  # Data frame that will contain the original plus the new model results.
  all <- results
  
  labels.order <- read.csv(paste0(dataset_path,"classes.csv"), header = F)[,1]
  
  df.v1 <- results[results$method == "v1",]
  df.v2 <- results[results$method == "v2",]
  
  for(i in 1:nrow(df.v1)){
    
    row.v1 <- df.v1[i,]
    row.v2 <- df.v2[i,]
    
    # Create arrays to store sum of scores and counts.
    sum.scores <- rep(0, length(labels.order))
    names(sum.scores) <- labels.order
    counts <- sum.scores
    
    ps1 <- strsplit(row.v1$predictionSet, "\\|")[[1]]
    if(length(ps1) == 0){
      ps1 <- ""
    }
    else{
      sum.scores[ps1] <- as.numeric(strsplit(row.v1$scores, "\\|")[[1]])
      counts[ps1] <- counts[ps1] + 1 
    }
    
    ps2 <- strsplit(row.v2$predictionSet, "\\|")[[1]]
    if(length(ps2) == 0){
      ps2 <- ""
    }
    else{
      sum.scores[ps2] <- sum.scores[ps2] + as.numeric(strsplit(row.v2$scores, "\\|")[[1]])
      counts[ps2] <- counts[ps2] + 1
    }
    
    
    # Compute average scores.
    avg.scores <- sum.scores / counts
    
    # Compute the intersection of the prediction sets.
    psInt <- intersect(ps1, ps2)
    
    
    if(length(psInt) > 0){
      tmp <- avg.scores[psInt]
      scores <- paste(tmp, collapse = "|")
    }
    else{
      scores <- ""
    }
    
    # Compute average p-values.
    pvs1 <- as.numeric(strsplit(row.v1$pvalues, "\\|")[[1]])
    pvs2 <- as.numeric(strsplit(row.v2$pvalues, "\\|")[[1]])
    
    avg.pvalues <- (pvs1 + pvs2) / 2
    
    avg.pvalues <- paste(avg.pvalues, collapse = "|")
    
    # Compute prediction.
    
    # If psInt != "" predict the one with highest score.
    # else take the max score from the union.
    if(length(psInt) > 0){
      prediction <- names(which.max(tmp))
    }
    else{
      prediction <- names(which.max(avg.scores))
    }
    
    if(is.null(prediction)){
      prediction <- ""
    }
    
    if(length(psInt) == 0){
      #psInt <- ""
      psInt <- prediction
    }
    else{
      psInt <- paste(psInt, collapse = "|")
    }
    
    # Build data frame row and add it.
    df <- data.frame(it=row.v1$it, method="mvc",
                     groundTruth=row.v1$groundTruth,
                     prediction=prediction,
                     predictionSet=psInt,
                     scores=scores,
                     pvalues=avg.pvalues)
    
    all <- rbind(all, df)
    
  } # end for
  
  write.csv(all, paste0(dataset_path,"results_1//results_mvc.csv"), row.names = F, quote = F)
  
}

summarize.iterations <- function(dataset_path, model.type){
  # Function to summarize the results per iteration.
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_",model.type,"//results_mvc.csv"))
  
  labels.order <- read.csv(paste0(dataset_path,"classes.csv"), header = F)[,1]
  
  methods <- unique(results$method)
  
  summary <- NULL
  
  for (i in sort(unique(results$it))){
    
    df.it <- results[results$it == i, ]
    
    for(m in methods){
      
      df <- df.it[df.it$method==m, ]
      
      # Compute classical performance metrics.
      all.labels <- unique(df$groundTruth)
      
      cm <- confusionMatrix(factor(df$prediction, levels = all.labels),
                            factor(df$groundTruth, levels = all.labels))
      
      acc <- cm$overall["Accuracy"]
      
      colavg <- colMeans(cm$byClass, na.rm = T)
      
      sensitivity <- colavg["Sensitivity"]
      
      specificity <- colavg["Specificity"]
      
      f1 <- colavg["F1"]
      
      # Compute conformal metrics.
      cvg <- coverage(df$groundTruth, df$predictionSet)
      
      setsize <- avg.set.size(df$predictionSet)
      
      pctempty <- pct.empty(df$predictionSet)
      
      OU <- observed.unconfidence(df$groundTruth, df$predictionSet, df$pvalues, labels.order)
      
      OE <- observed.excess(df$groundTruth, df$predictionSet)
      
      J <- jaccard(df$groundTruth, df$predictionSet)
      
      f.criterion <- average.fuzziness(df$pvalues)
      
      m.criterion <- pct.m.criterion(df$predictionSet)
      
      OM <- om.criterion(df$groundTruth, df$predictionSet)
      
      OF <- observed.fuzziness(df$groundTruth, df$predictionSet, df$pvalues, labels.order)
      
      tmp <- data.frame(it=i,
                        method=m, 
                        accuracy=acc,
                        sensitivity=sensitivity,
                        specificity=specificity,
                        F1=f1,
                        coverage=cvg,
                        jaccard=J,
                        setsize=setsize,
                        pctempty=pctempty,
                        MCriterion=m.criterion,
                        FCriterion=f.criterion,
                        OM=OM,
                        OF=OF,
                        OU,
                        OE,
                        row.names = NULL)
      
      summary <- rbind(summary, tmp)
    }
    
  }
  
  # Save results.
  write.csv(summary, 
            paste0(dataset_path,"results_",model.type,"//summary_iterations.csv"),
            row.names = FALSE,
            quote = FALSE)
  
  return(summary)
}

summarize.all <- function(dataset_path, model.type){
  
  # Read results.
  results <- read.csv(paste0(dataset_path,
                             "results_",model.type,"//summary_iterations.csv"))
  
  methods <- unique(results$method)
  
  summary <- NULL
  
  for(m in methods){
    df <- results[results$method==m,]
    
    # Compute means.
    avgs <- colMeans(df[,3:ncol(df)])
    names(avgs) <- paste0("mean_",names(avgs))
    
    # Compute standard deviations.
    sds <- apply(df[,3:ncol(df)],2,sd)
    names(sds) <- paste0("sd_",names(sds))
    
    # Build resulting data frame.
    tmp <- cbind(method=m, data.frame(t(avgs)), data.frame(t(sds)))
    
    summary <- rbind(summary, tmp)
  }
  
  # Save results.
  write.csv(summary, 
            paste0(dataset_path,"results_",model.type,"//summary_all.csv"),
            row.names = FALSE,
            quote = FALSE)
  
  return(summary)
  
}


pairwise.occurrences <- function(dataset_path, model.type, width=5, height=5){
  # Function to generate a plot of label co-occurrences in the sets.
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_",model.type,"//results_mvc.csv"))
  
  methods <- unique(results$method)
  
  #summary <- NULL
  
  for(m in methods){
    
    df <- results[results$method==m, ]
    
    all.labels <- sort(unique(df$groundTruth))
    
    ps <- strsplit(df$predictionSet, "\\|")
    
    n <- length(ps)
    
    # Create the co-occurrence matrix
    C <- matrix(data = rep(0, length(all.labels)^2),
                nrow = length(all.labels), 
                dimnames = list(row=all.labels, col=all.labels))
    
    
    for(i in 1:n){
      s <- ps[i][[1]]
      n2 <- length(s)
      if(n2 < 2)next;
      
      for(j in 1:(n2-1)){
        for(k in (j+1):n2){
           a <- s[j]; b <- s[k]
           C[a,b] <- C[a,b] + 1
           C[b,a] <- C[b,a] + 1
        }
      }
    }
    
    # Plot the matrix.
    noClasses <- length(all.labels)
    tamletras <- 12
    tamnums <- 3.5
    
    if(noClasses==20){
      tamnums <- 1.7
    }
    
    levels <- all.labels
    
    M <- C
    
    for(col in 1:noClasses){total <- sum(M[,col]);for(row in 1:noClasses){M[row,col] <- M[row,col] / total}}
    
    M[is.na(M)] <- 0.0
    
    z <- matrix(M, ncol=length(levels), byrow=TRUE, dimnames=list(levels,levels))
    confusion <- as.data.frame(as.table(z))
    
    plot1 <- ggplot()
    plot1 <- plot1 + geom_tile(aes(x=Var1, y=Var2, fill=Freq), data=confusion, color="black", linewidth=0.1) +
      ggtitle("co-occurrence matrix") +
      labs(x=expression(""), y=expression("")) +
      geom_text(aes(x=Var1,y=Var2, label=sprintf("%.3f", Freq)),data=confusion, size=tamnums, colour="black") +
      scale_fill_gradient(low="white", high="pink", name="counts")+
      theme(axis.text=element_text(colour="black", size =tamletras),
            axis.title = element_text(face="bold", colour="black", size=9),
            axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position="none", plot.margin = unit(c(.1,.1,.1,.1),"cm"), plot.title = element_text(size = 12, face = "bold", hjust = 0.5))
    
    pdf(paste0(dataset_path,"results_",model.type,"//c_",m,"_",model.type,".pdf"),
        width=width, height=height)
    print(plot1)
    dev.off()
    
  }
}

plot.confusion.matrix.aux <- function(dataset_path, model.type, width=5, height=5, diag0 = F){
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_",model.type,"//results_mvc.csv"))
  
  methods <- unique(results$method)
  
  for(m in methods){
    
    df <- results[results$method==m, ]
    
    all.labels <- sort(unique(df$groundTruth))
    
    
    M <- confusionMatrix(factor(df$prediction,all.labels),
                         factor(df$groundTruth,all.labels))$table
    n <- all.labels
    noClasses <- length(all.labels)
    tamletras <- 12
    tamnums <- 3.5
    
    if(noClasses==20){
      tamnums <- 1.7
    }
    
    # Set diagonal to 0
    if(diag0){
      diag(M) <- 0
    }
    
    for(col in 1:noClasses){total <- sum(M[,col]);for(row in 1:noClasses){M[row,col] <- M[row,col] / total}}
    
    M[is.na(M)] <- 0.0
    
    z <- matrix(M, ncol=noClasses, byrow=TRUE, dimnames=list(n,n))
    confusion <- as.data.frame(as.table(z))

    squarecolor <- "darkgreen"
    if(diag0){
      squarecolor <- "lightblue"
    }
    
    thetitle <- "confusion matrix"
    if(diag0 == TRUE){
      thetitle <- "zero diagonal confusion matrix"
    }
    
    plot1 <- ggplot()
    plot1 <- plot1 + geom_tile(aes(x=Var1, y=Var2, fill=Freq), data=confusion, color="black", linewidth=0.1) + 
      ggtitle(thetitle) +
      labs(x=expression(atop("True classes")), y=expression("Predicted classes")) +
      geom_text(aes(x=Var1,y=Var2, label=sprintf("%.3f", Freq)),data=confusion, size=tamnums, colour="black") +
      scale_fill_gradient(low="white", high=squarecolor, name="counts(%)")+
      theme(axis.text=element_text(colour="black", size =tamletras), 
            axis.title = element_text(face="bold", colour="black", size=9),
            axis.text.x = element_text(angle = 45, hjust = 1),
            legend.position="none", plot.margin = unit(c(.1,.1,.1,.1),"cm"), plot.title = element_text(size = 12, face = "bold", hjust = 0.5))
    
    fprefix <- "cm_"
    if(diag0){
      fprefix <- "cm0_"
    }
    pdf(paste0(dataset_path,"results_",model.type,"//",fprefix,m,"_",model.type,".pdf"),
        width=width, height=height)
    print(plot1)
    dev.off()
    
  }
}

plot.confusion.matrix <- function(dataset_path, model.type, width=5, height=5, diag0 = F){
  # This function calls the same function twice
  # to generate two confuson matrices. Normal and with Diag set to 0.
  plot.confusion.matrix.aux(dataset_path, model.type, width, height, diag0 = F)
  plot.confusion.matrix.aux(dataset_path, model.type, width, height, diag0 = T)
}

latex.summary <- function(dataset_path, model.type){
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_",model.type,"//summary_all.csv"))
  
  # Which metrics are to be multiplied by 100.
  #m <- c("accuracy","sensitivity","specificity","F1","coverage","pctempty")
  
  m <- c("accuracy|sensitivity|specificity|F1|coverage|pctempty")
  
  idxs <- grepl(m,colnames(results))
  
  results[,idxs] <- results[,idxs] * 100
  
  # Format to n decimal places.
  results[,2:ncol(results)] <- format(round(results[,2:ncol(results)], 2), nsmall = 2)
  
  # Merge mean and sd.
  n <- (ncol(results) - 1) / 2
  
  meancols <- results[,2:(2+n-1)]
  
  sdcols <- results[,(2+n):ncol(results)]
  
  for(i in 1:ncol(meancols)){
    meancols[,i] <- paste0(meancols[,i],"|",sdcols[,i])
  }
  
  # Format titles.
  titles <- gsub("mean_","",colnames(meancols))
  
  colnames(meancols) <- titles
  
  results <- cbind(method=results$method,meancols)
  
  final <- t(results)
  
  colnames(final) <- final[1,]
  final <- final[-1,]
  
  outfile <- paste0(dataset_path,"results_",model.type,"/tab_summary","_",model.type,".tex")
  
  out <- final %>%
    kbl(caption = paste0("Results ", model.type),
        format = "latex", booktabs = T) %>%
    kable_styling(latex_options = "striped")

    
  writeLines(out, outfile)
  
  
}

plot.scatter <- function(dataset_path){
  
  results <- read.csv(paste0(dataset_path,"results_1//summary_all.csv"))
  
  df <- results[,c(1,5,8)]
  
  #rownames(df) <- results$method
  
  p <- ggplot(df, aes(mean_setsize,mean_F1, color=method, show.legend = FALSE)) +
    geom_point(show.legend = FALSE) +
    geom_text_repel(aes(label = method), show.legend = FALSE, box.padding = .9) +
    labs(x = "set size", y = "F1 score")
  
  
  pdf(paste0(dataset_path,"results_1//scatter.pdf"), 8, 4)
  print(p)
  dev.off()
  
}

plot.histograms <- function(dataset_path, name){
  
  # Read results.
  results <- read.csv(paste0(dataset_path,"results_1//results_mvc.csv"))
  
  methods <- unique(results$method)
  
  for(m in methods){
    
    df <- results[results$method == m,]
    
    predictionSets <- df$predictionSet
    
    parts <- strsplit(predictionSets, "\\|")
    
    setsizes <- sapply(parts, function(x){
      length(x)
    })
    
    pdf(paste0(dataset_path,"results_1//",name,"_histogram_",m,".pdf"), 5, 3)
    
    barplot(table(setsizes)/length(setsizes), main = m)
    dev.off()
    
  }
  
}

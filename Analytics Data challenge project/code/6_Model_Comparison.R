# ---------------------------------------------------------------------------------------- #
#
# Model_Comparison.R 
# Author: Yesika Contreras
#  
# This code aims to compare the models that predicts `var0` using 
# the remaining columns from the data set contained in `braviant.csv`
#
# Imput are the models confusion Matrices
#
#
# R scripts generated 
# Sys.Date() # "2018-01-23"
#
# ---------------------------------------------------------------------------------------- #

#============================================================
# Libraries
#============================================================

library(Amelia)
library(dplyr)
library(corrplot)
library(ggplot2)
library(gridExtra)
library(car) 
library(caret)
library(e1071)
library(caTools)
library(randomForest)
library(pROC)
library(tidyr)


#============================================================
# Importing data
#============================================================

# To load the models
load("model_rf.RData")
load("model_rfUp.RData")
load("model_rfDown.RData")


#============================================================
# Comparison of models
#============================================================


#---------------------
# comparing AUC
#---------------------

plot (roc.rf1, col="darkgreen",main="ROC Statistical comparison")
lines(roc.rfDown, col="darkblue")
lines(roc.rfUp, col="red")
content <- roc.test(rocrf1, roc.rfDown,roc.rfUp)
text(50, 50, labels=paste(auc.rf1), adj=c(0, .5),col="darkgreen")
text(80, 63, labels=paste(auc.rfDown), adj=c(0, .5), col="darkblue")
text(90, 90, labels=paste(auc.rfUp), adj=c(0, .5), col="red")
legend("bottomright", legend=c( "RF","RF.DownSampling","RF.UpSampling"), col=c("darkgreen", "darkblue",'red'), lwd=2)




#---------------------
#comparing Accuracy
#---------------------

models <- list(rf_original = model.rf1,
               rf_down_sampling = model.rfDown,
               rf_up_sampling = model.rfUp)

resampling <- resamples(models)
bwplot(resampling, main= 'Accuracy comparison Random Forest', adj = 0)



#---------------------
# comparing : Precision, Recall, Sensitivity, Specificity 
#---------------------




models <- list(rf1 = model.rf1,
               rfDown = model.rfDown,
               rfUp = model.rfUp)


comparison <- data.frame(model = names(models))

for (name in names(models)) {
  model <- get(paste0("cm.", name))
  
  comparison[comparison$model == name, "Sensitivity"] <- model$byClass[["Sensitivity"]]
  comparison[comparison$model == name, "Specificity"] <- model$byClass[["Specificity"]]
  comparison[comparison$model == name, "Precision"] <- model$byClass[["Precision"]]
  comparison[comparison$model == name, "Balanced.Accuracy"] <- model$byClass[["Balanced Accuracy"]]
  
}

comparison %>%
  gather(x, y, Sensitivity:Balanced.Accuracy) %>%
  ggplot(aes(x = x, y = y, color = model, shape =model)) +
  geom_jitter(width = 0.2, alpha = 0.5, size = 5) +
  xlab('Metric') + ylab('Value') +
  ggtitle("Comparison metrics by Random Forest model")
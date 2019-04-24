# ---------------------------------------------------------------------------------------- #
#
# ModelRFDown.R 
# Author: Yesika Contreras
#  
# This code aims to fit a model that predicts `var0` using 
# the remaining columns from the data set contained in `braviant.csv`
#
#
# Model Random Forest with Down-sampling (Under-sampling)
# with both dataframes: numeric and numeric & categorical
# The input file is dataframes.RData file
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


#============================================================
# Importing data
#============================================================

# To load the dataframes 
load("~/dataframes.RData")


#============================================================
# Modeling
#============================================================

# Distribution data: Training:70%, test: 30% 
# Crossvalidation in training data: K= 10 folds
# Down-sampling (Under-sampling)

# ---------------------------------------------------
# For case 1: all variables numeric (dataframe: df)
# ---------------------------------------------------


# model.rfDown: model Random Forest with crossvalidation 
# ------------

set.seed(101)
a <- createDataPartition(df$var0, p = 0.7, list=FALSE)
train <- df[a,]
test <- df[-a,]

#k-fold Cross Validation, k=10 folds
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     savePredictions = TRUE, 
                     sampling = "down" )

# model
model.rfDown <- train(var0~., 
                   data=df, 
                   method="rf",  
                   ntree=200, 
                   importance = T, 
                   trControl=ctrl)

# print cv scores
print(model.rfDown)

plot(model.rfDown) #error rates

#prediction 
test$predict.rfDown = predict(model.rfDown, newdata=test)

pred = predict(model.rfDown, newdata=test)
accuracy <- table(pred, test[,'var0'])
sum(diag(accuracy))/sum(accuracy)
#[1] 0.7744313
misRate <- 1-sum(diag(accuracy))/sum(accuracy) ;misRate
#0.2255687
table(pred, test$var0) 
cm.rfDown <- confusionMatrix(data=pred, test$var0); cm.rfDown
#               Reference
#Prediction     0     1
#0 21377    15
#1  6411   685


# ROC curve
auc.rfDown  <- round(auc(test$var0, as.numeric(test$predict.rfDown)),4) ;auc.rfDown
#Area under the curve: 0.8742

roc.rfDown <- plot.roc(test$var0, as.numeric(test$predict.rfDown), main="ROC graph model.rDown", percent=TRUE, col="red")
text(30, 63, labels=paste(auc.rfDown), adj=c(0, .5))




#============================================================
# Saving Data
#============================================================


# Save model as RData
save(model.rfDown,cm.rfDown,auc.rfDown,roc.rfDown, file = "model_rfDown.RData")
# To load model again
load("model_rfUp.RData")


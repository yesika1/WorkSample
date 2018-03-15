# ---------------------------------------------------------------------------------------- #
#
# ModelRF.R 
# Author: Yesika Contreras
#  
# This code aims to fit a model that predicts `var0` using 
# the remaining columns from the data set contained in `braviant.csv`
#
#
# Model Random Forest
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
# crossvalidation in training data: K= 10 folds

# ---------------------------------------------------
# For case 1: all variables numeric (dataframe: df)
# ---------------------------------------------------


# model.rf1: model Random Forest with crossvalidation 
# ------------

set.seed(101)
a <- createDataPartition(df$var0, p = 0.7, list=FALSE)
train <- df[a,]
test <- df[-a,]

#k-fold Cross Validation, k=10 folds
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     savePredictions = TRUE )

# model
model.rf1 <- train(var0~., data=df, 
                   method="rf",  
                   ntree=200, 
                   importance = T, 
                   trControl=ctrl)

# print cv scores
print(model.rf1)

plot(model.rf1) #error rates

#prediction 
test$predict.rf1 = predict(model.rf1, newdata=test)

pred = predict(model.rf1, newdata=test)
accuracy <- table(pred, test[,'var0'])
sum(diag(accuracy))/sum(accuracy)
#[1] 0.9807638
misRate <- 1-sum(diag(accuracy))/sum(accuracy) ;misRate
# 0.01923617

table(pred, test$var0) 
cm.rf1 <- confusionMatrix(data=pred, test$var0); cm.rf1 
#               Reference
#Prediction     0     1
#0 27788   548
#1     0   152


# ROC curve
auc.rf1 <- round(auc(test$var0, as.numeric(test$predict.rf1)),4) ;auc.rf1 
#Area under the curve: 0.6079

roc.rf1 <- plot.roc(test$var0, as.numeric(test$predict.rf1), main="ROC graph model.rf1", percent=TRUE, col="red")
text(30, 63, labels=paste(auc.rf1), adj=c(0, .5))





# ------------------------------------------------------------------
# For case 2: variables numeric & categorical (dataframe: dfFactors)
# ------------------------------------------------------------------


# model.rf2: model Random Forest with crossvalidation 
# ------------

set.seed(101)
a <- createDataPartition(dfFactors$var0, p = 0.7, list=FALSE)
train <- dfFactors[a,]
test <- dfFactors[-a,]

#k-fold Cross Validation, k=10 folds
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     savePredictions = TRUE )

# model
model.rf2 <- train(var0~., 
                   data=dfFactors, 
                   method="rf",  
                   ntree=200, 
                   importance = T, 
                   trControl=ctrl)

# print cv scores
print(model.rf2)

plot(model.rf2) #error rates

#prediction 
test$predict.rf2 = predict(model.rf2, newdata=test)

pred = predict(model.rf2, newdata=test)
accuracy <- table(pred, test[,'var0'])
sum(diag(accuracy))/sum(accuracy)
#[1] 0.9756038
misRate <- 1-sum(diag(accuracy))/sum(accuracy) ;misRate
# 0.02439624
table(pred, test$var0)
cm.rf2 <- confusionMatrix(data=pred, test$var0) ;cm.rf2 
#               Reference
#Prediction     0     1


# ROC curve
auc.rf2  <- round(auc(test$var0, as.numeric(test$predict.rf2)),4) ;auc.rf2
#Area under the curve: 0.5036

roc.rf2 <- plot.roc(test$var0, as.numeric(test$predict.rf2), main="ROC graph model.rf2", percent=TRUE, col="cadetblue")
text(30, 63, labels=paste(auc.rf2), adj=c(0, .5))



#============================================================
# Saving Data
#============================================================


# Save model as RData
save(model.rf1,model.rf2,cm.rf1,cm.rf2, auc.rf1,roc.rf1,auc.rf2,roc.rf2, file = "model_rf.RData")
# To load model again
load("model_rf.RData")






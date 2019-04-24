# ---------------------------------------------------------------------------------------- #
#
# ModelLR.R 
# Author: Yesika Contreras
#  
# This code aims to fit a model that predicts `var0` using 
# the remaining columns from the data set contained in `braviant.csv`
# 
#
# Model  Logistic regression
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


# model.lr1: model logistic Regression with stepwise selection & crossvalidation
# ------------

set.seed(101)
a <- createDataPartition(df$var0, p = 0.7, list=FALSE)
train <- df[a,]
test <- df[-a,]


#k-fold Cross Validation, k=10 folds
ctrl <- trainControl(method = "cv", 
                     number = 10, 
                     savePredictions = TRUE)

# model
model.lr1 <- train(var0~., 
                   data=df, 
                   method="glmStepAIC", 
                   family=binomial(), 
                   trControl=ctrl)
#outcome  ~ var2 + var3 + var10 + var11 + var12
#AIC= 17725.99

# print cv scores
summary(model.lr1) 

#prediction 
test$predict.lr1 = predict(model.lr1, newdata=test)

pred = predict(model.lr1, newdata=test)
accuracy <- table(pred, test[,'var0'])
sum(diag(accuracy))/sum(accuracy)
# [1] 0.9755687
misRate <- 1-sum(diag(accuracy))/sum(accuracy) ;misRate
# 0.02443134
cm.lr1 <- confusionMatrix(data=pred, test$var0) ;cm.lr1 
#               Reference
#Prediction     0     1
#0 27771   672
#1    17    28


# ROC curve
auc.lr1 <- round(auc(test$var0, as.numeric(test$predict.lr1)),4) ;auc.lr1
#Area under the curve: 0.5147

roc.lr1 <- plot.roc(test$var0, as.numeric(test$predict.lr1), main="ROC graph model.lr1", percent=TRUE, col="red")
text(30, 63, labels=paste(auc.lr1 ), adj=c(0, .5))



# ------------------------------------------------------------------
# For case 2: variables numeric & categorical (dataframe: dfFactors)
# ------------------------------------------------------------------


# model.lr2: model logistic Regression with stepwise selection & crossvalidation
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
model.lr2 <- train(var0~., 
                   data=dfFactors, 
                   method="glmStepAIC", 
                   family=binomial(), 
                   trControl=ctrl)
#outcome ~ var21 + var22 + var23 + var24 + var25 + var26 + var27 + 
#var28 + var3 + var101 + var102 + var11
#AIC= 17727.41

# print cv scores
summary(model.lr2) 

#prediction 
test$predict.lr2 = predict(model.lr2, newdata=test)

pred = predict(model.lr2, newdata=test)
accuracy <- table(pred, test[,'var0'])
sum(diag(accuracy))/sum(accuracy)
## 0.9754634
misRate <- 1-sum(diag(accuracy))/sum(accuracy) ;misRate
# 0.02453665

cm.lr2 <- confusionMatrix(data=pred, test$var0)
#               Reference
#Prediction     0     1
#0 27771   682
#1    17    18

# ROC curve
auc.lr2 <- round(auc(test$var0, as.numeric(test$predict.lr2)),4) ;auc.lr2
#Area under the curve: 0.5126

roc.lr2 <- round(plot.roc(test$var0, as.numeric(test$predict.lr2), main="ROC graph model.lr2", percent=TRUE, col="cadetblue"),4)
text(30, 63, labels=paste(auc.lr2), adj=c(0, .5))



#============================================================
# Saving Data
#============================================================


# Save model as RData
save(model.lr1,model.lr2,cm.lr1,cm.lr2,auc.lr1,roc.lr1,auc.lr2,roc.lr2, file = "model_lr.RData")
# To load model again
load("model_lr.RData")

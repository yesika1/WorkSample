# ---------------------------------------------------------------------------------------- #
#
# ModelRFUp.R 
# Author: Yesika Contreras
# 
#
# Model Random Forest with Up-sampling (Under-sampling)
# with both dataframes: numeric and numeric & categorical
# The input file is dataframes.RData file
#
#
# This code aims to fit a model that predicts `var0` using 
# the remaining columns from the data set contained in `braviant.csv`
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
                     sampling = "up" )

# model
model.rfUp <- train(var0~., 
                   data=df, 
                   method="rf",  
                   ntree=200, 
                   importance = T, 
                   trControl=ctrl)

# print cv scores
print(model.rfUp)

plot(model.rfUp) #error rates

#prediction 
test$predict.rfUp = predict(model.rfUp, newdata=test)

pred = predict(model.rfUp, newdata=test)
accuracy <- table(pred, test[,'var0'])
sum(diag(accuracy))/sum(accuracy)
#[1] 1
misRate <- 1-sum(diag(accuracy))/sum(accuracy) ;misRate
# 0
table(pred, test$var0) 
cm.rfUp <- confusionMatrix(data=pred, test$var0); cm.rfUp
#               Reference
#Prediction     0     1
#Prediction     0     1
#0 27788     0
#1     0   700


# ROC curve
auc.rfUp  <- round(auc(test$var0, as.numeric(test$predict.rfUp)),4) ;auc.rfUp
#Area under the curve: 1

roc.rfUp <- plot.roc(test$var0, as.numeric(test$predict.rfUp), main="ROC graph model.rfUp", percent=TRUE, col="darkgreen")
text(30, 63, labels=paste(auc.rfUp), adj=c(0, .5))


#-------------------
# Model selected
#-------------------


# more information about the model
attributes(model.rfUp)

model.rfUp$trainingData
#       .outcome var1 var2 var3 var4 var5 var6 var8 var9 var10 var11 var12
#1             0    0    2  838  963    0    4    0    1     0   893     1
#2             0    2    1  837  593    1    1    0    4     1   389     1
#3             0    1    1  786  385    0    1    0    2     2   788     0
#Looking at the training data, the model did not vary the structure of the observations (no data with decimals for instance)

model.rfUp$resample
# Every sample has accuracy around  0.9737

model.rfUp$resampledCM
# Every sample has 9497 obs. where 9263 are class 0 and 234 are class 1

model.rfUp$times
#$everything
#user   system  elapsed 
#4019.547   41.599 4068.079 

#$final
#user  system elapsed 
#153.971   1.215 155.284 
varImp(model.rfUp)
#rf variable importance
#Importance
#var2     100.000
#var3      33.003
#var4      17.363
#var11     16.479
#var9      13.527
#var1       8.157
#var6       6.979
#var5       5.942
#var10      5.777
#var8       3.228
#var12      0.000

varImp(model.rfUp, scale = F)

model.rfUp$results


model.rfUp$bestTune #   mtry:    6
model.rfUp$finalModel
#No. of variables tried at each split: 6
#OOB estimate of  error rate: 0.18%



#============================================================
# Saving Data
#============================================================

# Save model as RData
save(model.rfUp,cm.rfUp,auc.rfUp,roc.rfUp, file = "model_rfUp.RData")
# To load model again
load("model_rfUp.RData")

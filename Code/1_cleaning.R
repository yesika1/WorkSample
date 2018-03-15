# ---------------------------------------------------------------------------------------- #
#
# Cleaning.R 
# Author: Yesika Contreras
#  
# This code generated the dataframes to be used in the modeling part
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


#============================================================
# Importing data
#============================================================

df <- read.csv("~/dataset.csv")
dfInitial <- read.csv("~/dataset.csv")
str(dfInitial)
View(df)
str(df) # 100000 obs. of  13 variables
head(df) # show first 6 observations
summary(df) # some var should  be factors, doubts about var9


#============================================================
# Cleaning data
#============================================================

# Removing '$' and transforming in numeric type
# --------------------------

df$var4 <- as.numeric( gsub('[$]','', df$var4 ,ignore.case = TRUE))
str(df)


#check for NA values
# --------------------------

sum(is.na(df)) # 0, but there are blanks in var8 

# var8: transform blanks in NA
print(table(df$var8))
#       N     Y 
# 4934 85537  9529 
prop.table(table(df$var8)) # 5% of data are blanks, will convert in NA

# Replacing blanks with NA in the df
df[df ==''] <- NA
df$var8 <- factor(df$var8)

# Var5: Replace outliers '9999' in NA
outlierVar5 <-(sum(df$var5 == 9999)) #11 obs.
(outlierVar5*100)/nrow(df) # 0.01% of data are 9999, will convert in NA
df[df ==9999] <- NA

# Var9: Replace outliers '-999' in NA
outlierVar5 <- (sum(df$var9 == -999)) #100 obs.
(outlierVar5*100)/nrow(df) # 0.1% of data are -999, will convert in NA
df[df ==-999] <- NA


## check for NA values
newNA <- sum(is.na(df)) # 5045
(newNA*100)/nrow(df)  # 5.045% will remove NA from dataset

#plotting NAs
pl1 <- missmap(df,y.at=c(1),y.labels = c(''),main = "Figure 1. Missing Data Map")

#Removing NAs from df
df <- df[!rowSums((is.na(df))),]
str(df)


# Note: Although NA= 5045, number of rows to remove are 5037, 8 observations presented 2 NA values in the same line

# Here we identified the the rows comparing with the original dataframe, to double check:
df2 <- dfInitial # making a dummy dataframe
df3 <-df2[ df2$var8==''& df2$var5==9999 & df2$var9 ==-999 ,]
nrow(df3)  #0
df3 <-df2[ df2$var8==''& df2$var5==9999 ,]
nrow(df3)  #1
df3 <-df2[ df2$var8==''& df2$var9 ==-999,] 
nrow(df3) #7
df3 <-df2[ df2$var5==9999 & df2$var9 ==-999,] 
nrow(df3) #0



#============================================================
# Feature Selection & Engineering
#============================================================

# Section composed by 2 cases to prepare the data for modeling
# first case: All variables considered as numeric
# Second case: Some variables are numeric and other categorical types


# ----------------------------------------------------
# CASE 1: dataframe called 'df'
# Transform all features as numeric type in the dataframe
# ----------------------------------------------------

# transforming levels of var8 & var10 in numeric
#-------
summary(df$var8)
#df$var8 <- factor(ifelse(df$var8 == "N", 0, 1))
df$var8 <- ifelse(df$var8 == "N", 0, 1) # as numeric 

summary(df$var10)
#df$var10 <- factor(ifelse(df$var10 == "A", 0, ifelse(df$var10 == "B",1,2)))
df$var10 <- ifelse(df$var10 == "A", 0, ifelse(df$var10 == "B",1,2))


# Dividing var4 by 100 to get numbers in similar range than var3
#-----------

df$var4 <- df$var4/100

# Transforming var0 into factor (target)
#------------
df$var0 <-factor(df$var0)
str(df)


# ----------------------------------------------------
# CASE 2: dataframe called 'dfFactors'
# Transforming some vars in factors from dataframe df
# ----------------------------------------------------

# To transform a factors: var8, var10 & var12  initially were factors, but we want same nomenclature in both dataframes
# Analyzing if the following features should be transformed: var1, var2, var5 & var6

# Analyzing var1
df2 <- df
df2$var1 <- factor(df2$var1)
summary(df2$var1) 
bar1 <-ggplot(df, aes(x=var1, color=var0)) + geom_bar(aes(fill=var0), position = 'fill') +ggtitle("Figure 7. Barplot var1 by factors") ; bar1
# proportion do not change considerable by level of target feature, will not be transformed in factor

# Analyzing var2
df2$var2 <- factor(df2$var2)
summary(df2$var2) 
bar2 <-ggplot(df, aes(x=var2, color=var0)) + geom_bar(aes(fill=var0), position = 'fill') +ggtitle("Figure 6. Barplot var2 by factors");bar2 
# proportion change by level of target feature, will be transformed into categorical feature

# Analyzing var5
df2$var5 <- factor(df2$var5)
summary(df2$var5) 
bar5 <-ggplot(df, aes(x=var5, color=var0)) + geom_bar(aes(fill=var0), position = 'fill');bar5 
# proportion dont change considerable by level of target feature, will not be transformed (similar graph to var1)

# Analyzing var6
df2$var6 <- factor(df2$var6)
summary(df2$var6) 
bar6 <-ggplot(df, aes(x=var6, color=var0)) + geom_bar(aes(fill=var0), position = 'fill') ;bar6
# proportion dont change considerable by level, will not be transformed

# conclusion, Features to transform into factors:  var2, var8, var10, var12  

# Transforming features into factors
dfFactors <- df
varNames <- c( 'var2','var8', 'var10', 'var12')
dfFactors[,varNames] <- lapply(dfFactors[,varNames] , factor)
str(dfFactors)



#============================================================
# Exploratory Data Analysis
#============================================================


str(df) #94963 obs. of  13 variables
summary(df)

#target variable
# ----------------------
tab <- table(df$var0)
propTable <- prop1 <- prop.table(tab)*100; 
print(propTable) # only 2.5% belong to label 1
#        0         1 
#97.542201  2.457799

pl1 <- ggplot(df, aes(var0)) + geom_bar( alpha=0.5,aes(fill=var0)) +ggtitle("Figure 4. Barplot Target Feature (var0)"); pl1

# Examine relationships among variables. 
#correlation: dependence or association okr any statitical relationship between two features

# Correlationfor dataframe df
# -----------------------

numCols <-sapply(df, is.numeric)
corData <- cor(df[,numCols])
corGraph <-corrplot(corData,method ='color', type='upper', addCoef.col="grey",order="hclust", title = 'Figure 2. Correlation Matrix',mar=c(0,0,1,0)) #Correlation matrix reordered according to the correlation coefficient. 
#Cleary we have very high negative correlation between var11 & var7 - var12 & var10 

#identify feature to remove due the correlation
# if two variables have a high correlation, the algorithm determines which one is involved with the most pairwise
#correlations and is removed.
x <- findCorrelation(corData, cutoff = 0.9) #7
#we should drop column 7 if we want to prevent multicollinearity, were column 7 is var7


# Correlation for dataframe dfFactors
#----------

numColsF <-sapply(dfFactors, is.numeric)
corDataF <- cor(dfFactors[,numColsF])
corGraphF <-corrplot(corDataF,method ='color', type='upper', addCoef.col="grey",order="hclust", title = 'Figure 2. Correlation Matrix',mar=c(0,0,1,0)) #Correlation matrix reordered according to the correlation coefficient. 
#Cleary we have very high negative correlation between var11 & var7 - var12 & var10 

#identify feature to remove due the correlation
# if two variables have a high correlation, the algorithm determines which one is involved with the most pairwise
#correlations and is removed.
xF <- findCorrelation(corDataF, cutoff = 0.9) #6
#we should drop column 6 if we want to prevent multicollinearity, were column 7 is var7

# scatterplot var7 vs. var11  #Here -1.0 states the perfect negative correlation between the variables.
pl2 <- ggplot(df, aes(x=var7, y=var11)) + geom_point(aes(shape=factor(var0), color=factor(var0)), size=4,alpha=0.2 ) +ggtitle("Figure 3. Scatterplot var7 by var11") #shape= for factors


# removing var7 from both dataframes
#----------

df$var7 <- NULL
dfFactors$var7 <- NULL

# more plots
#----------
# boxplot var4 and var0 : no distinction
pl3 <-ggplot(df, aes(var0,var4)) + geom_boxplot( aes( group=var0, fill=factor(var0), alpha=0.4)) +ggtitle("Figure 5. Boxplot var4 by Target Feature (var0)");pl3 

# boxplot var7 and var0 : no distinction
pl4 <-ggplot(df, aes(var0,var7)) + geom_boxplot( aes( group=var0, fill=factor(var0), alpha=0.4)) +ggtitle("Boxplot var7 by Target Feature (var0)") ;pl4

# boxplot var11 and var0 : no distinction
pl5 <-ggplot(df, aes(var0,var11)) + geom_boxplot( aes( group=var0, fill=factor(var0), alpha=0.4)) +ggtitle("Boxplot var11 by Target Feature (var0)") ;pl5

grid.arrange(pl1,pl3,bar2,bar1)
dev.off() # To reset the graphics device 


#============================================================
# Saving Data
#============================================================


# Save dataframes as RData
save(df, dfFactors, file = "dataframes.RData")
# To load the dataframes again
load("dataframes.RData")



# Setup the environment
# package.list<- c("knitr","ggplot2","dplyr","reshape2","ROCR","caTools","rpart",
#                  "rpart.plot","arules","scales")
# lapply(package.list, require, character.only = TRUE)
# options(scipen=6, width=100)

library(ggplot2)
#install.packages('caTools')
library(caTools)  #Used for data splitting

# Load dataset

credit_df = read.csv("/Users/jay/Downloads/Predicting Credit Card Approvals/datasets/cc_approvals.csv")

# Inspect data
tail(credit_df,17)

# Print summary statistics
credit_df_description = summary(credit_df)
print(credit_df_description)

# Print DataFrame information (View the structure of the data)
str(credit_df)

# Inspect missing values in the dataset
print(tail(credit_df))

# Count the number of NaNs in each column
colSums(is.na(credit_df))

#count the number of NaNS in dataframe
sum(is.na(credit_df))

# checking an example of '?' value
credit_df[674,]

# Replace the '?'s with NaN
credit_df[ credit_df == "?" ] <- NA
#Updating the levels of the factor variables 
credit_df[,-c(2:3,8,11,15)] <- lapply( credit_df[,-c(2:3,8,11,15)], factor )

# Count the number of NaNs in each column
colSums(is.na(credit_df))

#count the number of NaNS in dataframe
sum(is.na(credit_df))

# verify transformation
credit_df[674,]

## Imputation numerical variables

# Transforming Age into numerical value
credit_df$Age<-as.numeric(credit_df$Age)

## Imputation Age
mean_age<- mean(credit_df$Age,na.rm=T)

# Use correlation among numerical variables to predict missing age values
Numeric	<- credit_df[,c(2:3,8,11,15)]
colnames(Numeric)

round(cor(Numeric,use="complete.obs"),3)
#  The largest value in the first row is 0.395 meaning age is most closely correlated with YearsEmployed. 

age_imputate<-lm(Age~YearsEmployed, data=credit_df, na.action=na.exclude)
age_imputate$coefficients
age_missing_index<-which(is.na(credit_df$Age))
credit_df$Age[age_missing_index]<- predict(age_imputate,newdata=credit_df[age_missing_index,])

colSums(is.na(credit_df))

## Imputation Categorical variables

# tracking value before imputation
credit_df[674,]

# Generating a mode function were the input is called data and it is in the form df$column
mode <- function(data){
    val <- unique(data[!is.na(data)])  
    output <- val[which.max(tabulate(match(data, val)))] 
    return(output) 
}


# Using the apply function to run the mode function accross the columns in the dataframe
categorical_col <- c(1,4:7,9:10,12:14)
credit_df[categorical_col]<- lapply(credit_df[categorical_col],function(x) { x[is.na(x)] <- mode(x); x})


#credit_df$Gender[is.na(credit_df$Gender)] <- mode(credit_df$Gender) 
#df1[,subset1] <- as.data.frame(lapply(df1[,subset1],function(x) { x[is.na(x)] <- 0; x}

# tracking value after imputation
credit_df[674,]

colSums(is.na(credit_df))

# Convert binary values to 1 or 0
credit_df$Gender <- factor(ifelse(credit_df$Gender=="a",1,0))
credit_df$Employed <- factor(ifelse(credit_df$Employed=="t",1,0))
credit_df$PriorDefault<- factor(ifelse(credit_df$PriorDefault=="t",1,0))
credit_df$ApprovalStatus <- factor(ifelse(credit_df$ApprovalStatus=="+",1,0))

str(credit_df)

# Drop the features 'DriversLicense', 'ZipCode'
credit_df = credit_df[-c(12,14)]

# Split data in training a testing datasets
library(caTools)
set.seed(123)
split<- sample.split(credit_df$ApprovalStatus, SplitRatio=0.75)
Train<- subset(credit_df,split==TRUE)
Test <- subset(credit_df, split==FALSE)

# New dataframes shape
dim(Train)
dim(Test)

# Get success rates in training set
table(Train$ApprovalStatus)

# We dont need to rescale for R

# Create logrithmic regresion model = base model
logreg<- glm(ApprovalStatus~., data=Train,family=binomial)

summary(logreg)

# Apply the model to the test set
logreg_predict<-predict(logreg, newdata=Test,type="response")


# Create a confusion Matrix
table(Test$ApprovalStatus,logreg_predict>0.5)

# Use the step function to simplify the model with a function
backwards = step(logreg) # Backwards selection is the default

formula(backwards)

summary(backwards)

# Apply the model to the test set
backwards_predict<-predict(backwards, newdata=Test,type="response")


# Create a confusion Matrix
table(Test$ApprovalStatus,backwards_predict>0.5)



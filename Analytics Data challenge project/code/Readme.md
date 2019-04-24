
# Data Analysis project:
Fitting a model that predicts `var0` using the remaining columns from the data set contained in `dataset.csv`.

The R code is provided with the step by step procedure to generate the proposed models, distributed in the following files:
1_cleaning.R: Cleaning and Exploratory Data Analysis, the output of this file is the dataframes.RData file.
2_modelLR.R: A Logistic Regression Model is generated here
3_modelRF.R: A Random Forest Model is generated here
4_modelRFDown.R: A Random Forest Model with down-sampling is generated here
5_modelRFUp.R: A Random Forest Model with up-sampling is generated here
6_modelComparison.R: Comparison among Random Forest Model and Random Forest Model with down-smpling and up-sampling


Input file:
Dataset.csv

Corresponding outputs:
dataframes.RData: cleaned dataframes
model_lr: Logistic Regression model output 
model_rf: Random Forest model output
model_rfDown: Random Forest Model with down-sampling model output
model_rfUp: Random Forest Model with up-sampling model output


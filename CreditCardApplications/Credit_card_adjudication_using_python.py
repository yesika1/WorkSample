#!/usr/bin/env python
# coding: utf-8

# ## Project: Credit Card Application Approvals using Python
# This notebook contains a credit card approval predictor for commercial banks using machine learning techniques.
# 
# 
# ### Dataset
# <p>For this project, the dataset  was extracted from the UCI Machine Learning Repository <a href="http://archive.ics.uci.edu/ml/datasets/credit+approval">Credit Card Approval dataset</a>
# 
# The dataset contains data for 690 customers that applied for credit with a retail bank. There are 16 attributes captured for each customer; including a decision flag which allows you to identify those customers which were approved and denied for credit.
# 
# ### Summary
# The analysis of this project consist on the creation of a model to evaluate the decision to approve or deny credit card applications. The final model created is a logarithmic regression model. This model was able to predict the outcome of a credit applications with 84% accuracy which was significantly better performance than the baseline model.
# 
# As a conclusion, there are four drivers that possitively affect the approval decision, as these factors increase, so does the probability that a credit card will be issued.
# 
# 
# Applications can get rejected for many reasons, including, like high loan balances, low income levels, or too many inquiries on an individual's credit report, among others. The four influencing factors are:
# 
# Prior default,
# Years employed,
# Credit score, and
# Income level.
# Other variables such as age, sex, or ethnicity did not have an influence on whether the application was denied. A Chi Squared test for independence validated our conclusion Ethnicity and Approval status are independent.
# 
# 
# ### Notebook' structure
# <ul>The structure of this notebook is as follows:
# 
# <li>First, loading and viewing the dataset.</li>
# <li>Second, preprocessing the dataset to ensure the machine learning model we choose can make good predictions.</li>
# <li>Third, doing some exploratory data analysis to build our intuitions.</li>
# <li>Finally, we will build a machine learning model using Logistic Regression that can predict if an individual's application for a credit card will be accepted.</li>

# ## 1. Loading and viewing the dataset
# <p> We find that since this data is confidential, the dataset has alterations on the original data.</p>

# # 2. Inspecting the applications
# Inspecting the structure, numerical summary, and specific rows of the dataset.
# - the dataset has a mixture of numerical and non-numerical features. This can be fixed with some preprocessing.
# - Specifically, the features 2, 7, 10 and 14 contain numeric values (of types float64, float64, int64 and int64 respectively) and all the other features contain non-numeric values.
# - The dataset also contains values from several ranges. Some features have a value range of 0 - 28, some have a range of 2 - 67, and some have a range of 1017 - 100000. 
# - We can get useful statistical information (like <code>mean</code>, <code>max</code>, and <code>min</code>) about the features that have numerical values. 
# 

# In[16]:


# Print summary statistics
credit_df_description = credit_df.describe()
print(credit_df_description)

print("\n")

# Print DataFrame information
credit_df_info = credit_df.info()
print(credit_df_info)

print("\n")

# Inspect missing values in the dataset
print(credit_df.tail(17))


# ## 3. Handling missing values (Marking missing values as NaN)
# 
# Marking Missing Values or corrupted data as NaN. Then, we can count the number of true values in each column.
# 
# - The dataset has missing values. The missing values in the dataset are labeled with '?'.
# - Let's temporarily replace these missing value question marks with NaN.
# - A total of 67 missing values were identified

# In[17]:


# Import numpy
import numpy as np

# Inspect missing values in the dataset
print(credit_df.tail(17))

# Count the number of NaNs in each column
print(credit_df.isnull().sum())

# Replace the '?'s with NaN
credit_df = credit_df.replace('?', np.nan)

# Inspect the missing values again
print(credit_df.tail(17))

# Count the number of NaNs in each column
print(credit_df.isnull().sum())


# ## 5. Handling the missing values (Data Imputation)
# Median Imputation for numerical data and Frequent value for categorical data.
# 
# - There are not missing values for numerical variables.
# 
# - There are still some missing values to be imputed for columns Gender, Age, Married, BankCustomer, EducationLevel, Ethnicity and ZipCode. All of these columns contain non-numeric data and we are going to impute these missing values with the most frequent values as present in the respective columns. 

# In[18]:


# Iterate over each column of credit_df
for col in credit_df.columns:
    # Check if the column is of object type
    if credit_df[col].dtypes == 'object':
        # Impute with the most frequent value
        credit_df = credit_df.fillna(credit_df[col].value_counts().index[0])

# Count the number of NaNs in the dataset and print the counts to verify
print(credit_df.isnull().sum())


# ## 6. Preprocessing the data (Convert the non-numeric values to numeric)
# 
# we will be converting all the non-numeric values into numeric ones using label encoding. 
# 
# We do this because not only it results in a faster computation but also many machine learning models (like XGBoost and especially the ones developed using scikit-learn) require the data to be in numeric format. 

# In[19]:


# Import LabelEncoder
from sklearn import preprocessing

# Instantiate LabelEncoder
le = preprocessing.LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in credit_df.columns:
    # Compare if the dtype is object
    if credit_df[col].dtypes =='object':
    # Use LabelEncoder to do the numeric transformation
        credit_df[col]=le.fit_transform(credit_df[col])
    print(le.classes_)
    


# ## 7. Splitting the dataset into train and test sets and Feature selection
# 
# <p>Now, we will split our data into train and test sets. 
# Ideally, no information from the test data should be used to scale the training data or should be used to direct the training process of a machine learning model. Hence, we first split the data and then apply the scaling.
# <p>Also, features like <code>DriversLicense</code> and <code>ZipCode</code> are not as important as the other features in the dataset for predicting credit card approvals. We should drop them to design our machine learning model with the best set of features.

# In[25]:


# Import train_test_split
from sklearn.model_selection import train_test_split

# Drop the features 11 and 13 
credit_df = credit_df.drop(['DriversLicense', 'ZipCode'], axis=1)

# Convert the DataFrame to a NumPy array
credit_df = credit_df.values

# Segregate features and labels into separate variables
X,y = credit_df[:,0:12] , credit_df[:,13]

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X,
                                y,
                                test_size=0.25,
                                random_state=42)
print(X_train.shape)
print(X_test.shape)


# ## 8. Preprocessing the data (Rescaling Data to an uniform range)
# 
# We are only left with one final preprocessing step of scaling data between 0-1 before we can fit a machine learning model to the data.
# 
# For example, the credit score, CreditScore, of a person is their creditworthiness based on their credit history. The higher this number, the more financially trustworthy a person is considered to be. So, a CreditScore of 1 is the highest since we're rescaling all the values to the range of 0-1.

# In[29]:


# Import MinMaxScaler
from sklearn.preprocessing import MinMaxScaler

# Instantiate MinMaxScaler and use it to rescale X_train and X_test
scaler = MinMaxScaler(feature_range=(0,1))
rescaledX_train = scaler.fit_transform(X_train)
rescaledX_test = scaler.fit_transform(X_test)1

rescaledX_train[:1]


# ## 9. Fitting a logistic regression model to the train set
# <p>Essentially, predicting if a credit card application will be approved or not is a <a href="https://en.wikipedia.org/wiki/Statistical_classification">classification</a> task. <a href="http://archive.ics.uci.edu/ml/machine-learning-databases/credit-screening/crx.names">According to UCI</a>, our dataset contains more instances that correspond to "Denied" status than instances corresponding to "Approved" status. Specifically, out of 690 instances, there are 383 (55.5%) applications that got denied and 307 (44.5%) applications that got approved. </p>
# <p>This gives us a benchmark. A good machine learning model should be able to accurately predict the status of the applications with respect to these statistics.</p>
# <p>Which model should we pick? A question to ask is: <em>are the features that affect the credit card approval decision process correlated with each other?</em> they indeed are correlated. Because of this correlation, we'll take advantage of the fact that generalized linear models perform well in these cases. Let's start our machine learning modeling with a Logistic Regression model (a generalized linear model).</p>

# In[37]:


# Import LogisticRegression
from sklearn.linear_model import LogisticRegression

# Instantiate a LogisticRegression classifier with default parameter values
logreg = LogisticRegression(solver='lbfgs')

# Fit logreg to the train set
logreg.fit(rescaledX_train,y_train )


# ## 10. Making predictions and evaluating performance
# But how well does our model perform?
# 
# We will now evaluate our model on the test set with respect to classification accuracy. But we will also take a look the model's confusion matrix. In the case of predicting credit card applications, it is equally important to see if our machine learning model is able to predict the approval status of the applications as denied that originally got denied. If our model is not performing well in this aspect, then it might end up approving the application that should have been approved. The confusion matrix helps us to view our model's performance from these aspects.
# 
# - Our model was pretty good! It was able to yield an accuracy score of almost 84%.</p>
# - For the confusion matrix, the first element of the of the first row of the confusion matrix denotes the true negatives meaning the number of negative instances (denied applications) predicted by the model correctly. And the last element of the second row of the confusion matrix denotes the true positives meaning the number of positive instances (approved applications) predicted by the model correctly.</p>

# In[38]:


# Import confusion_matrix
from sklearn.metrics import confusion_matrix

# Use logreg to predict instances from the test set and store it
y_pred = logreg.predict(rescaledX_test)

# Get the accuracy score of logreg model and print it
print("Accuracy of logistic regression classifier: ", logreg.score(rescaledX_test, y_test))

# Print the confusion matrix of the logreg model
confusion_matrix(y_test, y_pred) # y_true Vs y_pred


# ## 11. Grid searching and making the model perform better
# 
# <p>Let's see if we can do better. We can perform a <a href="https://machinelearningmastery.com/how-to-tune-algorithm-parameters-with-scikit-learn/">grid search</a> of the model parameters to improve the model's ability to predict credit card approvals.</p>
# <p><a href="http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html">scikit-learn's implementation of logistic regression</a> consists of different hyperparameters but we will grid search over the following two:</p>
# <ul>
# <li>tol</li>
# <li>max_iter</li>
# </ul>

# In[39]:


# Import GridSearchCV
from sklearn.model_selection import GridSearchCV

# Define the grid of values for tol and max_iter
tol = [0.01, 0.001, 0.0001]
max_iter = [100, 150, 200]

# Create a dictionary where tol and max_iter are keys and the lists of their values are corresponding values
param_grid = dict(tol=tol, max_iter=max_iter)


# ## 12. Finding the best performing model
# We have defined the grid of hyperparameter values and converted them into a single dictionary format which GridSearchCV() expects as one of its parameters. Now, we will begin the grid search to see which values perform best.
# 
# We will instantiate GridSearchCV() with our earlier logreg model with all the data we have. Instead of passing train and test sets separately, we will supply X (scaled version) and y. We will also instruct GridSearchCV() to perform a cross-validation of five folds.
# 
# We'll end the notebook by storing the best-achieved score and the respective best parameters.
# 
# While building this credit card predictor, we tackled some of the most widely-known preprocessing steps such as scaling, label encoding, and missing value imputation. We finished with some machine learning to predict if a person's application for a credit card would get approved or not given some information about that person.

# In[40]:


# Grid searching is a process of finding an optimal set of values for the parameters of a certain machine learning model.

# Instantiate GridSearchCV with the required parameters
grid_model = GridSearchCV(estimator=logreg, param_grid=param_grid, cv=5)

# Use scaler to rescale X and assign it to rescaledX
rescaledX = scaler.fit_transform(X)

# Fit data to grid_model
grid_model_result = grid_model.fit( rescaledX, y)

# Summarize results
best_score, best_params = grid_model_result.best_score_, grid_model_result.best_params_
print("Best: %f using %s" % (best_score, best_params))


# In[ ]:





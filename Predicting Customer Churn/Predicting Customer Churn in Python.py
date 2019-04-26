#!/usr/bin/env python
# coding: utf-8

# ## Predicting Customer Churn in Python
# 
# Churn quantifies the number of customers who have unsubscribed or canceled their service contract.
# It is very expensive to win them back once lost, additionally they will not do the best word to mouth marketing if unsatisfied. 
# 
# We look at data from customers that already have churned (response) and their characteristics / behavior (predictors) before the churn happened. By fitting a statistical model that relates the predictors to the response, we will try to predict the response for existing customers
# 
# 
# ## The Dataset
# We take an available dataset you can find on IBMs retention programs: Telcom Customer Churn Dataset. The raw dataset contains more than 7000 entries and 21 features. All entries have several features and of course a column stating if the customer has churned or not.
# 
# The data set includes information about:
# 
# Customers who left within the last month – the column is called Churn
# Services that each customer has signed up for – phone, multiple lines, internet, online security, online backup, device protection, tech support, and streaming TV and movies
# Customer account information – how long they’ve been a customer, contract, payment method, paperless billing, monthly charges, and total charges
# Demographic info about customers – gender, age range, and if they have partners and dependents
# 
# ## Conclusions
# 
# According with the findings of the Logistic Regression model that has an 80% accuracy, we noticed that the following features are the strongest key drivers:
# 
# - Features that having then increase the probability of a customer to churn are PaperlessBilling and SeniorCitizen.
# 
# - Features that having then decrease the probability of a customer to churn are Contract,PhoneService, TechSupport, OnlineSecurity, and Dependents 

# ## 1. Loading and viewing the dataset
# 
# - Each row represents a customer, each column contains customer’s attributes described on the column Metadata.
# - The raw data contains 7043 rows (customers) and 21 columns (features).
# - The “Churn” column is our target.
# - We see that 26,5% Of the total amount of customer churn. 

# In[109]:


## Import packages

import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from pylab import rcParams
get_ipython().run_line_magic('matplotlib', 'inline')


# Loading the CSV with pandas
data = pd.read_csv('/Users/jay/Downloads/WA_Fn-UseC_-Telco-Customer-Churn.csv')

print(data.info())
data.head(5)


# In[110]:


# Plotting the target variable
sizes = data['Churn'].value_counts(sort = True)
colors = ["grey","purple"] 
rcParams['figure.figsize'] = 5,5

# Plot
labels= ['No', 'Yes']
plt.pie(sizes, labels=labels, colors=colors,
        autopct='%1.1f%%', shadow=True, startangle=270,)
plt.title('Percentage of Churn in IBM Dataset')
plt.show()


# ## 2. Data Preparation and Feature Engineering
# ### Dropping irrelevant data
# 
# There may be data included that is not needed to improve our results. 
# Best is that to identify by logic thinking or by creating a correlation matrix. 
# In this data set we have the customerID for example. As it does not influence our predicted outcome, we drop it.
# 

# In[111]:


## Dropping CustomerID
data.drop(['customerID'], axis=1, inplace=True)


# ### Handle Missing Values
# The values can be identified by the “.isnull()” function in pandas for example. 
# After identifying the null values it depends on each case if it makes sense to fill the missing value for example with the mean, median or the mode, or in case there is enough training data drop the entry completely. 
# 
# The dataset does not present null values.

# In[112]:


## Identify missing values
data.isnull().sum()
#pd.Series([np.nan]).sum()


# ### Converting Numerical Features From Object (Label Encoding)
# - we can see that the the column TotalCharges are numbers, but actually in the object format. Our machine learning model can only work with actual numeric data. Therefore with the “to_numeric” function we can change the format and prepare the data for our machine learning model

# In[113]:


data.dtypes


# In[121]:


# Converting 'TotalCharges' into numerical value 

# I used 0 to replace anything that isn't a number
data['TotalCharges'] = pd.to_numeric(data['TotalCharges'], errors='coerce').fillna(0).astype(float)

# float(df['a'][1]) # works for one index
# data['TotalCharges'].astype(dtype=np.float64)
# data['TotalCharges'].astype(float)
# pd.to_numeric(data['TotalCharges']) # Unable to parse string " " at position 488


# In[122]:


data.isnull().sum()


# In[123]:


## Converting categorical data into numerical data

# Import LabelEncoder
from sklearn import preprocessing

# Instantiate LabelEncoder
le = preprocessing.LabelEncoder()

# Iterate over all the values of each column and extract their dtypes
for col in data.columns:
    # Compare if the dtype is object
    if data[col].dtypes =='object':
    # Use LabelEncoder to do the numeric transformation
        data[col]=le.fit_transform(data[col])
    print(le.classes_)
    


# In[125]:


data.head(2)


# ### 3. Splitting the dataset
# 
# First our model needs to be trained, second our model needs to be tested. Therefore it is best to have two different dataset. As for now we only have one, it is very common to split the data accordingly. X is the data with the independent variables, Y is the data with the dependent variable. The test size variable determines in which ratio the data will be split. It is quite common to do this in a 80 Training / 20 Test ratio.

# In[129]:


data["Churn"] = data["Churn"].astype(int)
y = data["Churn"].values
X = data.drop(labels = ["Churn"],axis = 1)

# Create Train & Test Data
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=101)

print(X_train.shape, X_test.shape, y_train.shape, y_test.shape )


# ## 4. Logistic Regression & Model Testing
# 
# Logistic Regression is one of the most used machine learning algorithm and mainly used when the dependent variable (here churn 1 or churn 0) is categorical. 
# - Step 1. Let’s Import the model we want to use from sci-kit learn
# - Step 2. We make an instance of the Model
# - Step 3. Is training the model on the training data set and storing the information learned from the data

# In[131]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
result = model.fit(X_train, y_train)


# In[132]:


from sklearn import metrics
prediction_test = model.predict(X_test)
# Print the prediction accuracy
print (metrics.accuracy_score(y_test, prediction_test))


# The score show us that in 80% of the cases our model predicted the right outcome for our binary classification problem. That’s considered quite good for a first run, 

# **Finding the independent variables have to most influence on our predicted outcome**
# 
# So with the final objective to reduce churn and take the right preventing actions in time, we want to know which independent variables have to most influence on our predicted outcome. Therefore we set the coefficients in our model to zero and look at the weights of each variable.

# In[133]:


# To get the weights of all the variables
weights = pd.Series(model.coef_[0],
 index=X.columns.values)
weights.sort_values(ascending = False)


# A positive value has a positive impact on our predicted variable. A good example is “SeniorCitizen”: The positive relation to churn means that having this type of contract also increases the probability of a customer to churn. On the other hand that “PhoneService” is in a highly negative relation to the predicted variable, which means that customers with this type of contract are very unlikely to churn. 

# In[ ]:





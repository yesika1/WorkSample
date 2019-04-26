## Predicting Customer Churn in Python

The following project consists on identify and quantifies the number of customers who have unsubscribed or canceled their service contract.
We use the available dataset on IBMs retention programs from Kaggle: Telcom Customer Churn Dataset

* First, Load and explore the dataset

* Second, we preprocessed data: handling missing values, dropping irrelevant data, transforming data type (label encoding)

* Finally, we generated a Logistic Regression model to find the key drivers on customer churning.


### Conclusions
* According with the findings of the Logistic Regression model that has an 80% accuracy, we noticed that the following features are the strongest key drivers:

- Features that having then increase the probability of a customer to churn are PaperlessBilling and SeniorCitizen.

- Features that having then decrease the probability of a customer to churn are Contract,PhoneService, TechSupport, OnlineSecurity, and Dependents 

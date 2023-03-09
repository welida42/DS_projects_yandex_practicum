# Predicting mobile operator customers churn

#### Project objectives: 
- To predict the churn of mobile operator's customers in order to plan predictive measures (discounts, special conditions).
Personal data, information about tariffs and contracts are available for customer sampling.
- Build a model with the value of the quality metric ROC_AUC not less than 0.85.
Additionally fix the model's accuracy.
#### Project tasks:
Exploratory data analysis
- Data uploading
- Data preprocessing
- Data analysis and feature selection

Model selection and training
- Logistic Regression
- Random Forest
- Gradient Busting (lightgbm)

Prediction Results Analysis


#### Results: 

The analysis of signs was performed and the following insights were revealed.
- With the increase in monthly payments the number of clients who left increases
- In the first months the number of leased clients is high, but with time the number of leased clients decreases.
- A big part of quitting clients are non-cash payment users.

The following models were trained: 
- LogisticRegression
- RandomForestClassifier
- LGBMClassifier

The best quality of predictions was obtained with LGBMClassifier. ROC_AUC = 0.91 on the test sample.

The greatest contribution to the model prediction is made by the attributes:
- duration_days - number of days the customer uses the services
- MonthlyCharges - monthly payment for services. This indicator is higher for the users who left.
- TotalCharges - the total amount that the client paid for the whole time of using the services

#### Used Libraries:
- python pandas numpy sklearn matplotlib seaborn lightgbm catboost


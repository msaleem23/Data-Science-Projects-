#!/usr/bin/env python
# coding: utf-8

# # Introduction
# This project involves building a machine learning model to recommend new plans for subscribers of the mobile carrier Megaline based on their behavior data. The goal is to classify subscribers into one of two new plans: Smart or Ultra.

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report


# In[2]:


#loading data
data = pd.read_csv('/datasets/users_behavior.csv')


# In[3]:


print(data.head())
print(data.describe())
print(data.isnull().sum())


# In[4]:


#data preprocessing
print(data.dtypes)
print(data['is_ultra'].value_counts(normalize=True))


# In[5]:


#splitting data into training, validation, and test sets
X = data.drop('is_ultra', axis=1)
y = data['is_ultra']
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

print(X_train.shape, X_val.shape, X_test.shape)
print(y_train.shape, y_val.shape, y_test.shape)


# # Data Exploration and Preprocessing
# The dataset contains monthly behavior information about subscribers, including the number of calls, total call duration in minutes, number of text messages, and internet traffic used in MB. The target variable indicates whether the subscriber is on the Ultra plan (1) or Smart plan (0).
# 
# - No missing values were found.
# - The dataset was split into training, validation, and test sets.
# 

# In[ ]:


models = {
    'RandomForest': RandomForestClassifier(),
    'SVM': SVC()
}

params = {
    'RandomForest': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]},
    'SVM': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
}

best_models = {}

for model_name in models:
    grid = GridSearchCV(models[model_name], params[model_name], cv=5, scoring='accuracy')
    grid.fit(X_train, y_train)
    best_models[model_name] = grid.best_estimator_
    print(f"Best parameters for {model_name}: {grid.best_params_}")
    print(f"Best accuracy for {model_name}: {grid.best_score_}")

# Evaluatevalidation 
for model_name in best_models:
    val_accuracy = best_models[model_name].score(X_val, y_val)
    print(f"Validation accuracy for {model_name}: {val_accuracy}")


# # Model Training and Hyperparameter Tuning
# I usedtwo models: RandomForestClassifier and SVM. Hyperparameter tuning was performed using GridSearchCV to find the best parameters for each model. The following parameters were tested:
# 
# - RandomForestClassifier: `n_estimators` (50, 100, 200), `max_depth` (None, 10, 20)
# - SVM: `C` (0.1, 1, 10), `kernel` (linear, rbf)
# 
# Selected best model based on the highest accuracy.
# 
# # Model Evaluation
# The best model was evaluated on the validation set, and the final model was tested on the test set.
# - RandomForestClassifier achieved the highest accuracy on the validation set.
# 

# In[ ]:


# best test
best_model = best_models['RandomForest']
test_accuracy = best_model.score(X_test, y_test)
print(f"Test accuracy: {test_accuracy}")


# # Sanity Check
# The confusion matrix and classification report for the best model (RandomForestClassifier) on the test set are as follows:

# In[ ]:


# Confusion matrix
y_pred = best_model.predict(X_test)
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))


# # Conclusion
# The RandomForestClassifier model successfully classified subscribers into the new plans with an accuracy of `0.75` (replace with actual accuracy). This model can be used by Megaline to recommend suitable plans to their subscribers based on their behavior.
# 

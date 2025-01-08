# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 14:09:37 2025

@author: avina
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Load your dataset
df_er = pd.read_csv('P:/My Documents/Books & Research/Analytics Vidya Blackbelt program/Foundational ML Algorithms I/modified_employee_turnover.csv')
df_er.columns
df_er.dtypes
df_er.shape
df_er.head()
missing_values = df_er.isnull().sum()
print(missing_values)

# Define features (X) and target (y)
X = df_er.drop(columns=['Employee_Turnover'])  
y = df_er['Employee_Turnover']  

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Normalize the data for logistic regression
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Logistic regression model
log_reg = LogisticRegression(max_iter=500, random_state=42)
log_reg.fit(X_train_scaled, y_train)
y_pred = log_reg.predict(X_test_scaled)
print("Logistic Regression without regularization:\n", classification_report(y_test, y_pred))


# Regularization strengths to test
Cs= np.linspace(0.1,20,50)

# Logistic Regression with L1 Regularization
logistic_l1_cv = LogisticRegressionCV(
    Cs=Cs,
    penalty='l1',
    solver='liblinear',
    cv=5,
    max_iter=10000,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
logistic_l1_cv.fit(X_train_scaled, y_train)
y_pred_l1 = logistic_l1_cv.predict(X_test_scaled)
print("L1 Regularization:\n", classification_report(y_test, y_pred_l1))

# Logistic Regression with L2 Regularization
logistic_l2_cv = LogisticRegressionCV(
    Cs=Cs,
    penalty='l2',
    solver='liblinear',
    cv=5,
    max_iter=10000,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
logistic_l2_cv.fit(X_train_scaled, y_train)
y_pred_l2 = logistic_l2_cv.predict(X_test_scaled)
print("L2 Regularization:\n", classification_report(y_test, y_pred_l2))

# Logistic Regression with Elastic Net Regularization
logistic_en_cv = LogisticRegressionCV(
    Cs=Cs,
    penalty='elasticnet',
    solver='saga',
    l1_ratios=[0.5],
    cv=5,
    max_iter=10000,
    scoring='f1',
    n_jobs=-1,
    random_state=42
)
logistic_en_cv.fit(X_train_scaled, y_train)
y_pred_en = logistic_en_cv.predict(X_test_scaled)
print("Elastic Net Regularization:\n", classification_report(y_test, y_pred_en))

# Plotting Model Complexity
regularization_types = ['L1', 'L2', 'Elastic Net']
models = [logistic_l1_cv, logistic_l2_cv, logistic_en_cv]
complexity = {}

for reg, model in zip(regularization_types, models):
    complexity[reg] = {
        'C': model.C_[0],
        'Accuracy': cross_val_score(model, X_train_scaled, y_train, cv=5, scoring='accuracy').mean()
    }

# Visualize Model Complexity
plt.figure(figsize=(10, 6))
for reg in regularization_types:
    plt.bar(reg, complexity[reg]['Accuracy'], label=f"C={complexity[reg]['C']:.2f}")

plt.title("Model Complexity vs Accuracy for Regularization Types")
plt.xlabel("Regularization Type")
plt.ylabel("Cross-Validated Accuracy")
plt.legend()
plt.grid()
plt.show()

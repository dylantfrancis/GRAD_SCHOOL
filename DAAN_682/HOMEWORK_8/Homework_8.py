# ========S2=====================================================================
# DAAN: 682: Data Analytics Programming in Python
# Author: Dylan Francis
# Title: Homework_8: Supervised Learning with Scikit-learn:classification Models 
# =============================================================================
# 
# Quantitative	Attributes
# Age	(years)
# BMI	(kg/m2)
# Glucose	(mg/dL)
# Insulin	(µU/mL)
# HOMA	
# Leptin	(ng/mL)
# Adiponectin	(µg/mL)
# Resistin	(ng/mL)
# MCP-1(pg/dL)	(ng/mL)
# Labels
# 1 = Healthy Controls
# 2 = Patients

# Homework Questions: 
#1.) Perform data exploratory data analysis (EDA) and interpret the results: display the data's basic information and summary statistics, perform data quality checks, examine the distribution of the target variable, and visualize the feature distributions. Note: A comprehensive EDA should provide the essential foundation for data-driven decision-making when building predictive models later. (10 points)
#2.) Use 30% of the data as the test set and build a Logistic regression model to predict the Classification variable. (20 points)
#3.) Build a Naïve Bayes model to predict the Classification variable. (20 points)
#4.) Build a Decision tree model to predict the Classification variable. (20 points)
#5.) Build a Neural network model to predict the Classification variable. (20 points)
#6.) Which model is the best? Which variable is the most important one? (10 points)
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn import datasets  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB

path = r"C:\Users\dylan\OneDrive\Documents\GRAD_SCHOOL\DAAN_682\HOMEWORK_8"
target = "Classification"
os.chdir(path)
breastcancer = pd.read_csv("breastcancer.csv")

#1.) Data exploration
print(f"\nThere are {breastcancer.shape[0]} rows, and {breastcancer.shape[1]} columns.")
print("\n=======DATASET_INFO======")
print(breastcancer.info())
print("\n=======SUMMARY_STATS======")
print(breastcancer.describe())
breastcancer_corr = breastcancer.corr()
print(f"\n Here is the correlation martrix: {breastcancer_corr}")
print("\n===== MISSING VALUES =====")
print(f"{breastcancer.isnull().sum()}")
print("\n===== DUPLICATE ROWS =====")
print(f"The total number of duplicate rows is: {breastcancer.duplicated().sum()}")

# Here is the plot for the target variable (classification)
plt.figure(figsize=(6,4))
counter = sns.countplot(x=breastcancer[target])
for container in counter.containers:
    counter.bar_label(container)
plt.title("Target Variable Distribution")
plt.show()

# Here is the visualization of the feature variables
for col in breastcancer.columns:
    if(col != target):
        plt.figure(figsize=(6,4))
        sns.histplot(breastcancer[col], kde=True)
        plt.title(f"Feature Variable, {col} Distribution")
        plt.xlabel(col)
        plt.ylabel("Count")
        plt.show()
        
#2.) 
features = breastcancer.drop(columns=[target])
targets = breastcancer[target]

features_train, features_test, targets_train, targets_test = train_test_split(
    features, targets, test_size=0.30, random_state=42
)

print(f"\nTraining set size: {features_train.shape}")
print(f"Test set size: {features_test.shape}")

log_model = LogisticRegression(max_iter=1000)
log_model.fit(features_train,targets_train)
targets_pred = log_model.predict(features_test)


print("\n MODEL PERFORMANCE")
print(f"Accuracy: {accuracy_score(targets_test, targets_pred)}")

print("\n CONFUSION MATRIX ")
print(confusion_matrix(targets_test, targets_pred))

print("\n CLASSIFICATION REPORT ")
print(classification_report(targets_test, targets_pred))

#3.) 
# Initialize the Naïve Bayes model
nb_model = GaussianNB()

# Train the model
nb_model.fit(features_train, targets_train)

# Make predictions
targets_pred_nb = nb_model.predict(features_test)

# Evaluate the model
print("\n===== NAIVE BAYES MODEL PERFORMANCE =====")
print(f"Accuracy: {accuracy_score(targets_test, targets_pred_nb)}")

print("\n===== CONFUSION MATRIX =====")
print(confusion_matrix(targets_test, targets_pred_nb))

# ========S2=====================================================================
# DAAN: 682: Data Analytics Programming in Python
# Author: Dylan Francis
# Title: Homework_10: Unsupervised Learning with SciKit Learn III: Adcanced Models 
# =============================================================================
# 
# =============================================================================
# Perform exploratory data analysis on the Breast Cancer Data and interpret the results. (10 points)
# Build and evaluate SVM (Support Vector Machine) models using different kernel functions. Report the performance for each kernel and interpret the results. (40 points)
# Build Random Forest models with different values of n_estimators. Report the accuracy for each value and determine the optimal n_estimators. (25 points)
# Build AdaBoost models with different values of n_estimators. Report the accuracy for each value and determine the optimal n_estimators. (25 points)
# =============================================================================

import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier

path = r"C:\Users\dylan\OneDrive\Documents\GRAD_SCHOOL\DAAN_682\HOMEWORK_8"
os.chdir(path)
breastcancer = pd.read_csv("breastcancer.csv")
target = "Classification"

# =============================================================================
#1.) Data exploratory
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

# =============================================================================
#2.) 
features = breastcancer.drop(columns=[target])
targets = breastcancer[target]
scaler = StandardScaler()

#split the data into 80% training and 20% test
x_train, x_test, y_train, y_test = train_test_split(
    features, targets, test_size=0.2, random_state=13, stratify=targets)

#scale the data so no data "weighs" too heavily
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

#linear model
svm_linear = svm.SVC(kernel = 'linear')
svm_linear.fit(x_train_scaled, y_train)

#rbf model
svm_rbf = svm.SVC(kernel = 'rbf', gamma = 'scale')
svm_rbf.fit(x_train_scaled, y_train)

# Poly model
svm_poly = svm.SVC(kernel = 'poly', degree =3)
svm_poly.fit(x_train_scaled, y_train)

# Predications
predictions = {
    "Linear_SVM": svm_linear.predict(x_test_scaled),
    "RBF_SVM": svm_rbf.predict(x_test_scaled),
    "Poly_SVM": svm_poly.predict(x_test_scaled)
}

# Results 
results = []

for name, preds in predictions.items():
    print(f"\nCLASSIFICATION REPORT: {name}")
    print(classification_report(y_test, preds))
    
print("""The classification report suggest very similar results for the Linear and RBF models.
      These two models yield roughly the same precision, recall, F-1 scores, and accuracy. The
      model that is distinctly different, and produces worse performance on this data set, is the poly model,
      which performed worse across the board. """)

# =============================================================================
#3.) random forest

# I started with a wide range of n_values, [10, 50, 100, 200, 400, 800, 1000].
# I literated on this list until I got in the range of the highest accuracy.
n_values = list(range(1,31))
rf_results = []

for n in n_values:
    rf_model = RandomForestClassifier(n_estimators=n, random_state = 42)
    rf_model.fit(x_train, y_train)
    preds = rf_model.predict(x_test)
    
    accuracy = metrics.accuracy_score(y_test, preds)
    rf_results.append({
        "n_estimators": n,
        "Accuracy": accuracy
        })
    
rf_results_df = pd.DataFrame(rf_results)
print("\n===== Random Forest Summary =====")
print(rf_results_df)
print("""I iterated on the number of trees several time to determine the optimal number of estimators. 
      Several numbers produced the highest accurary of 83%. The number of Ns that yeilded this result was 13-26,
      with an exception of 23, which had a small drop.""")
      
# =============================================================================
#4.) AdaBoost models 
n_values_boots = list(range(1,51))
boost_results = []

for n in n_values_boots:
    boost_model = AdaBoostClassifier(n_estimators=n, random_state=42)
    boost_model.fit(x_train_scaled, y_train)
    boost_model.predict(x_test_scaled)
    preds_boost = boost_model.predict(x_test_scaled)
    
    boost_acc = metrics.accuracy_score(y_test, preds_boost)
    boost_results.append({
        "n_estimators":n,
        "Accuracy": boost_acc
        })
boost_results_df = pd.DataFrame(boost_results)
print("\n===== AdaBoost Summary =====")
print(boost_results_df)
print("""I get the highest accuracy when n = 16, 18, and 19. The model appears to drop in accuracy after that,
      which is largely expected due to the weight given the misclassifcation errors.""")
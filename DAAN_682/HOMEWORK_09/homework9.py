# ========S2=====================================================================
# DAAN: 682: Data Analytics Programming in Python
# Author: Dylan Francis
# Title: Homework_9: Supervised Learning with Scikit-learn II: regression Models 
# =============================================================================
# 
# =============================================================================
# 
# Perform exploratory data analysis and interpret the results. Remove the 'motor_UPDRS' column as it should not be used as a predictor. (10 points)
# Use cross-validation to build a Linear Regression model to predict total_UPDRS. Report and analyze the results using different metrics such as MAE, MSE, and R2. (25 points)
# Use cross-validation to build a Regression Tree model to predict total_UPDRS. Report and analyze the results using different metrics such as MAE, MSE, and R2. (25 points)
# Use cross-validation to build a Neural Network model to predict total_UPDRS. Report and analyze the results using different metrics such as MAE, MSE, and R2. (25 points)
# Compare their performance with MAE and MSE, which model has better performance? Propose at least one specific technique to potentially improve the performance. (5 points)
# Choose either the Regression Tree model or Neural Network model and optimize it by tuning its parameters. Report and analyze how these parameters help improve the performance. (10 points)
# =============================================================================

import os
import pandas as pd
import numpy as np
from sklearn import datasets  
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import KFold, cross_validate
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
  
path = r"C:\Users\dylan\OneDrive\Documents\GRAD_SCHOOL\DAAN_682\HOMEWORK_9\parkinsons+telemonitoring"
os.chdir(path)
df = pd.read_csv("parkinsons_updrs.data")
  

#1.) Exploratory analysis and interpreting results. 
print(f"\nThere are {df.shape[0]} rows, and {df.shape[1]} columns in the features dataframe")
print("\n=======DATASET_INFO======")
print(df.info())
print("\n=======SUMMARY_STATS======")
print(df.describe())
print("\n===== MISSING VALUES =====")
print(f"{df.isnull().sum()}")
print("\n===== DUPLICATE ROWS =====")
print(f"The total number of duplicate rows is: {df.duplicated().sum()}")

df_modified = df.drop(["motor_UPDRS", "subject#"], axis=1)
print(df_modified)
print(f"\nThere are {df_modified.shape[0]} rows, and {df_modified.shape[1]} columns in the features dataframe")
print("\n=======DATASET_INFO======")
print(df_modified.info())
print("\n=======SUMMARY_STATS======")
print(df_modified.describe())
print("\n===== MISSING VALUES =====")
print(f"{df_modified.isnull().sum()}")

plt.figure(figsize=(12, 10))
df_corr = df_modified.corr()
sns.heatmap(df_corr, cmap='coolwarm', center=0)
plt.title("Correlation Matrix")
plt.show()

# Targets
fig, ax = plt.subplots(figsize=(12, 5))
sns.histplot(df_modified['total_UPDRS'], kde=True, ax=ax)
plt.show()
print("Based on the histogram plot of the target plot, the data is approximately normal, potentiallly slight right skew.")
print("if the data is right skewed, then that would suggest most patients have lower UPDs scores, with fewer high-severity cases.")


corr = df_modified.corr()['total_UPDRS'].sort_values()
corr.plot(kind='barh', figsize=(8,10))
plt.title("Feature Correlation with total_UPDRS")
plt.show()
print("This grahph shows the features displayed from most positvely correlated to most negatively correlated.")

#2.) getting into the actaul modeling now (regression)
kf = KFold(n_splits=5, shuffle=True, random_state=42)

x = df_modified.drop(columns=['total_UPDRS'])
y = df['total_UPDRS']

lin_reg_model = LinearRegression()
scoring = {'MAE': 'neg_mean_absolute_error', 'MSE': 'neg_mean_squared_error', 'R2': 'r2'}

results = cross_validate(lin_reg_model, x, y, cv=kf, scoring=scoring)

# Convert negative errors to positive (since higher is "better")
lr_mae = -results['test_MAE']
lr_mse = -results['test_MSE']
lr_r2 = results['test_R2']

print("\n \n Linear Regression Results ")
print("MAE:", lr_mae)
print("MSE:", lr_mse)
print("R2:", lr_r2)

print("\n Here are the average results for Linear Regression")
print("Mean MAE:", np.mean(lr_mae))
print("Mean MSE:", np.mean(lr_mse))
print("Mean R2:", np.mean(lr_r2))

print(f"\n The mean MAE is {np.mean(lr_mae)}, which is a fairly large error, suggesting low prediction accuracy.")
print(f"\n The mean MSE is {np.mean(lr_mse)}, which means the model is making large prediction errors, suggesting some outliers could exisit.")
print(f"\n A R^2 of {np.mean(lr_r2)} suggests only  {np.mean(lr_r2)*100} % of the total variance of the target is predicted by the model.")

#3.) Decision tree:
tree_model = DecisionTreeRegressor(random_state=42)
tree_results = cross_validate(tree_model, x, y, cv=kf, scoring=scoring)

# Convert negative errors to positive (since higher is "better")
tree_mae = -tree_results['test_MAE']
tree_mse = -tree_results['test_MSE']
tree_r2 = tree_results['test_R2']

print("\n \n Regression Tree Results ")
print("MAE:", tree_mae)
print("MSE:", tree_mse)
print("R2:", tree_r2)

print("\n Average Results (Regression Tree)")
print("Mean MAE:", np.mean(tree_mae))
print("Mean MSE:", np.mean(tree_mse))
print("Mean R2:", np.mean(tree_r2))

print(f"\n The mean MAE is {np.mean(tree_mae)}, which is a very small error, suggesting high prediction accuracy. This is predicting UPDRS within 1 unit")
print(f"\n The mean MSE is {np.mean(tree_mse)}, which is pretty small, suggesting very few extreme outliers. There is strong consistenicy in the model's predictions.")
print(f"\n A R^2 of {np.mean(tree_r2)} suggests {np.mean(tree_r2)*100} % of the total variance of the target is predicted by the model. This is very high, and nearly all of the variance in UPDRS is explained by the model")
print("\n The regession tree performs better than linear regression in all metrics examined.")

#4.)
nn_model = MLPRegressor(random_state=42, max_iter=500)
nn_results = cross_validate(nn_model, x, y, cv=kf, scoring=scoring)

# Convert negative errors to positive (since higher is "better")
nn_mae = -nn_results['test_MAE']
nn_mse = -nn_results['test_MSE']
nn_r2 = nn_results['test_R2']

print("\n \n Neural Network Results ")
print("MAE:", nn_mae)
print("MSE:", nn_mse)
print("R2:", nn_r2)

print("\n Average Results (Neural Network)")
print("Mean MAE:", np.mean(nn_mae))
print("Mean MSE:", np.mean(nn_mse))
print("Mean R2:", np.mean(nn_r2))

print(f"\n The mean MAE is {np.mean(nn_mae)}, which is a large error, suggesting low prediction accuracy. This is predicting UPDRS within ~8 units")
print(f"\n The mean MSE is {np.mean(nn_mse)}, which is large, suggesting extreme outliers. There is not strong consistenicy in the model's predictions.")
print(f"\n A R^2 of {np.mean(nn_r2)} suggests {np.mean(nn_r2)*100} % of the total variance of the target is predicted by the model. This is very low.")


#5.) comparing all the results

results_summary = pd.DataFrame({
    "Model": ["Linear Regression", "Regression Tree", "Neural Network"],
    "MAE": [np.mean(lr_mae), np.mean(tree_mae), np.mean(nn_mae)],
    "MSE": [np.mean(lr_mse), np.mean(tree_mse), np.mean(nn_mse)],
    "R2": [np.mean(lr_r2), np.mean(tree_r2), np.mean(nn_r2)]
})

print(results_summary)

print("\n Based off the summary table, we can safely conclude that Regression Tree is the best mocdel for this data set and objective.")
print("\n It produces the lowest MAE and MSE and the highest R^2.")
print("\n The Linear Regression model marginally out performs the neural network model but ultimtley the results are about the same, and signficantly less accurate than the Regression Tree.")
print("\n One way to potentially improve the Regression Tree is to limit how deep or complex the tree can go. With the current implmentation, there is the risk that the model is overfitting the data.")

#6.) The regression tree is performaing very well and I doubt I will have much imporvement so I will focus on the neural network. Optimizing the neural network through scaling. 

nn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('nn', MLPRegressor(max_iter=1000, random_state=42))
])

param_grid_nn = {
    'nn__hidden_layer_sizes': [(20,), (20,20)],
    'nn__activation': ['relu'],
    'nn__alpha': [0.001],
    'nn__learning_rate': ['adaptive']
}

grid_search_nn = GridSearchCV(
    nn_pipeline,
    param_grid_nn,
    cv=kf,
    scoring='neg_mean_absolute_error',
    n_jobs=-1
)

grid_search_nn.fit(x, y)

best_nn_model = grid_search_nn.best_estimator_

nn_opt_results = cross_validate(best_nn_model, x, y, cv=kf, scoring=scoring)

nn_opt_mae = -nn_opt_results['test_MAE']
nn_opt_mse = -nn_opt_results['test_MSE']
nn_opt_r2 = nn_opt_results['test_R2']

print("\n \n Optimized Neural Network Results ")
print("MAE:", nn_opt_mae)
print("MSE:", nn_opt_mse)
print("R2:", nn_opt_r2)

print("\n Average Results (Optimized Neural Network)")
print("Mean MAE:", np.mean(nn_opt_mae))
print("Mean MSE:", np.mean(nn_opt_mse))
print("Mean R2:", np.mean(nn_opt_r2))

results_summary = pd.DataFrame({
    "Model": ["Linear Regression", "Regression Tree", "Neural Network", "Optimized NN"],
    "MAE": [np.mean(lr_mae), np.mean(tree_mae), np.mean(nn_mae), np.mean(nn_opt_mae)],
    "MSE": [np.mean(lr_mse), np.mean(tree_mse), np.mean(nn_mse), np.mean(nn_opt_mse)],
    "R2": [np.mean(lr_r2), np.mean(tree_r2), np.mean(nn_r2), np.mean(nn_opt_r2)]
})
print(results_summary)
print("\n The neural network has improved, by decreasing Mean MAE and MSE while increasing R^2.")
print("\n There was one warning that popped up while running the optimized neural network - Maximum iterations (1000) reached and the optimization hasn't converged yet.")
print("I believe a convergence would be reached if I increased the max iterations, however that significantly increased run time. ")
# ========S2=====================================================================
# DAAN: 682: Data Analytics Programming in Python
# Author: Dylan Francis
# Title: Homework_3: Statistical Analysis with Pandas 
#
# 
# Homework Questions:
# 1.) Import data mtcars.csv into Python. (10 points)
# 
# 2.) Explore the data and perform a statistical analysis of the data. (30 points)
# 
# 3.) Analyze mpg for cars with different gear, and show your findings. (20 points)
# 
# 4.) Analyze mpg for cars with different carb, and show your findings. (20 points)
# 
# 5.) Find out which attribute has the most impact on mpg. (20 points)
# =============================================================================
import numpy as np
import pandas as pd
from pandas import Series, DataFrame
import os

#1. 
#importing and reading the data
path = r"C:\Users\dylan\OneDrive\Documents\GRAD_SCHOOL\DAAN_682\HOMEWORK_3"
os.chdir(path)
mtcars = pd.read_csv("mtcars.csv")

#2
#high level overview of the data before conducting the statistical analysis
print("A high level overview of the data is listed below: \n")
mtcars.info()

#summary statistics
print(f"A summary of statistics of mtcars are listed below. These summary stats include: mean, std., min, max, and quartiles. \n {mtcars.describe()}")

results = []
for column in mtcars:
    max_idx = mtcars[column].idxmax()
    max_value = mtcars[column].max()
    model_name = mtcars.loc[max_idx, "model"]
    results.append({"Column": column, "Model": model_name, "Max_value": max_value})
print("Here are the max values for each of the indexes, along with the corresponding model")
print(pd.DataFrame(results))


#finding the highest correlation for each variable 
correlation_matrix = mtcars.drop(columns="model").corr()
print(f"The correlation matrix is given by: {correlation_matrix}")

for col in correlation_matrix.columns:
    row = correlation_matrix[col].drop(col)
    top_var = row.idxmax()
    top_corr = row[top_var]
    print(f"{col} has the highest positive correlation with {top_var}, with a correlation of {top_corr}")
    
#3 
group_by_variable_3 = 'gear'
mpg_stats = mtcars.groupby(group_by_variable_3)['mpg'].agg(['mean', 'median', 'min', 'max'])
print(mpg_stats)

counter = pd.Series(0, index=mpg_stats.index)
for col in mpg_stats.columns:
    idx = mpg_stats[col].idxmax()
    counter[idx]+=1
print(f"The {mpg_stats.index.name} with that produces the best MPG is {mpg_stats.index.name}: {counter.idxmax()}")

#4
group_by_variable_4 = 'carb'
mpg_stats = mtcars.groupby(group_by_variable_4)['mpg'].agg(['mean', 'median', 'min'])
print(mpg_stats)

counter = pd.Series(0, index=mpg_stats.index)
for col in mpg_stats.columns:
    idx = mpg_stats[col].idxmax()
    counter[idx]+=1
print(f"The {mpg_stats.index.name} with that produces the best MPG is {mpg_stats.index.name}: {counter.idxmax()}")

#5 
# this should be the similar to the calcualtion in number 2 excpet I need to account for absoutle value
correlation_edits = mtcars.drop(columns=['model']).corr().abs()['mpg'].sort_values(ascending=False)

print(f"The attribute that contributes the most to mpg is {correlation_edits.index[1]} with a correlation value of: {correlation_edits.iloc[1]}")


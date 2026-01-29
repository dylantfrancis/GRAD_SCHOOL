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
summary_stats = mtcars.describe()

#high level overview of the data before conducting the statistical analysis
print("A high level overview of the data is listed below: \n")
print(mtcars.info())

print(f"Summary_stats of mtcars are listed below. These summary stats include:mean, std., min, max, and quartiles. \n {summary_stats}")
results = []
for column in mtcars.columns:
    max_idx = mtcars[column].idxmax()
    max_value = mtcars[column].max()
    model_name = mtcars.loc[max_idx, "model"]
    results.append({"Column": column, "Model": model_name, "Max_value": max_value})
    
print(pd.DataFrame(results))
#print(f"The location of min values for each of the columns is: {mtcars.min()} and is located at: {mtcars.idxmin()}")
#print(f"The location of max values for each of the columns is located at: {mtcars.idxmax()}")



#5. 
#removing the first row and first column
mtcars_edited = mtcars.drop("model", axis=1)
correlation = mtcars_edited.corr()
#print(f"the correlation martix is given by: {correlation}")

most_corr ={}
corr_value ={}

for row in correlation.index:
    row_corr =correlation.loc[row].drop(row)
    max_var = correlation.idxmax()
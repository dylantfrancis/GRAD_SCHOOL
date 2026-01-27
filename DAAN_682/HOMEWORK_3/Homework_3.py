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


mtcars = pd.read_csv("mtcars.csv")
print(mtcars)
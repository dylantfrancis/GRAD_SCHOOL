# ========S2=====================================================================
# DAAN: 682: Data Analytics Programming in Python
# Author: Dylan Francis
# Title: Homework_4: Data Cleaning, Processing, and Manipulating with Pandas
#
# 
# Homework Questions:
#1.1 Explore the datasets. (10 points)
#1.2 Find and handle missing values in the data. (It is your choice how you handle the missing data.) ( 20 points)
#1.3 Explore the variable column and convert the "variable” column to dummy variables and join the dummies to the data. (20 points)
#1.4 Convert the "one” column into 3 bins. (20 points)import numpy as np

# s = “I am happy to join with you today in what will go down in history as the greatest demonstration for freedom in the history of our nation. Five score years ago, a great American, in whose symbolic shadow we stand today, signed the Emancipation Proclamation. This momentous decree came as a great beacon light of hope to millions of Negro slaves who had been seared in the flames of withering injustice. It came as a joyous daybreak to end the long night of their captivity. But one hundred years later, the Negro still is not free. One hundred years later, the life of the Negro is still sadly crippled by the manacles of segregation and the chains of discrimination. One hundred years later, the Negro lives on a lonely island of poverty in the midst of a vast ocean of material prosperity. One hundred years later, the Negro is still languishing in the corners of American society and finds himself an exile in his own land. So we have come here today to dramatize a shameful condition."
#2.1 Find out how many unique words in s. (10 points)
#2.2 Which word appears the most? (10 points)
#2.3 How many words start with ‘t’ (case-insensitive). (10 points).

import pandas as pd
from pandas import Series, DataFrame
import os

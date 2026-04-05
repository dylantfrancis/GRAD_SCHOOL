#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 18 07:57:11 2026

@author: dylanfrancis
"""

# 1.1) Use the randn function to create an array with a dimension of 5X5 and use a for loop to calculate the sum of all elements in the diagonal of the array. (25 points)
# 1.2) Choose any three functions to apply to this array. (25 points)


# 2.1) Use x = np.random.randint(0, 1000, size = (10, 10)) to generate 10x10 array and use a for loop to find out how many even numbers are in it. (25 points)
# 2.2) Randomly generate an 8x9 array from a normal distribution with mean = 1,
# sigma = 0.5. Calculate the mean of elements whose indexes have a relation of 
# (i+j)%5 == 0  (i is row index and j is column index).

import numpy as np

# 1.1) 
print("Problem: 1.1\n")
arrary_size=5
test_arrary = np.random.randn(arrary_size,arrary_size)
print(test_arrary, "\n")

sum_diag=0
for i in range(arrary_size):
        sum_diag += test_arrary[i,i]
        print(i)

print("The sum of the diagonals of the array is: ", sum_diag, "\n")
#print("Doublechecking the sum of the diagnoals: ", np.trace(test_arrary))
      
# 1.2)
print("Problem: 1.2\n")
print("The sum of the elements in the .randn matrix is:", np.sum(test_arrary), "but their mean is: ", np.mean(test_arrary), "\n")

print("The standard deviations of each column of the .randn matrix are:", np.std(test_arrary, axis=0), "\n")

print("The location of the min number in the flatted .randn matrix is: ", np.argmin(test_arrary),
      "and the location of the max number in the flatted .randn matrix is:", np.argmax(test_arrary), "\n")

# 2.1) 
print("Problem: 2.1\n")
x = np.random.randint(0, 1000, size = (10, 10))
print(x)

even_counter = 0
for row in x:
    for value in row:
        if value %2 ==0:
            even_counter+=1
        
print(f"There are {even_counter} even numbers in this array \n")        
        
#2.2)
print("Problem: 2.2\n")
array = np.random.normal(loc=1.0, scale=0.5, size=(8, 9))

x=0
value_sum =0
for i, row in enumerate(array):
    for j, value in enumerate(row):
        if (i+j) % 5 == 0:
            x+=1
            value_sum+=value
mean = value_sum / x
print(f"\n The sum of the values at the indexes whose sum is divisible by 5 is: {mean}")
            


#code provided by the professor for question 1: 
import numpy as np

L1 = []

np.random.seed(56)

for i in np.random.randint(0, 100, 10):
    L1.extend([i] * np.random.randint(0, 100, 1)[0])

np.random.shuffle(L1)

L2 = [879, 394, 235, 580, 628, 81, 206, 238, 927, 853, 622, 603, 110, 143, 824, 324, 343, 506, 634, 325, 258, 900, 960, 286, 449, 890, 921, 170, 888, 851]



#Homeworkquestions listed below: 
# 1.1) What are the unique values? (5 points)
unique_values = list(set(L1))
print("\n The unique values in the list are:", (unique_values), "\n")


# 1.2) How many unique values? (5 points)
print("The number of unique values in L1 is:", (len(unique_values)),"\n")

# 1.3) Create a dictionary with the unique items in L1 as dictionary keys and their count as the dictionary values. (20 points)
mydict = {}
for x in L1:
    mydict[x] = mydict.get(x, 0) + 1        
print("Here is a dictionary with unique items from L1 as the keys and the count as the values", mydict, "\n")
print()
      
# 1.4) Which value appears most frequently? The manual comparison is not acceptable. (10 points)
most_frequent_value = max(mydict, key=mydict.get)
most_frequent_count = mydict[most_frequent_value]
print("Most frequent value:", most_frequent_value, "Most frequent count:", most_frequent_count, "\n")

# 2.1) Use a while loop to calculate the sum of the even numbers in L2. (10 points)
even_sum=0
odd_sum=0
j=0 
while j < len(L2):
    if L2[j] % 2 ==0:
        even_sum=even_sum+L2[j]
    else:
        odd_sum=odd_sum+L2[j]
    j+=1
print("The sum of the even numbers is: ",even_sum, "\n")
    
# 2.2) Write a function to calculate the mean of a list. Use this function to calculate the mean of L2 (10 points)
def mean_calculator(x):
    total =0
    for value in x:
        total=total+value
    mean = total / len(x)
    return mean
print("The mean of L2 is:" ,mean_calculator(L2), "\n")

# 2.3) Calculate the sum for elements in L2 which is larger than 500. (10 points)
sum_greater_500 =0
for k in L2:
    if k > 500:
        sum_greater_500+=k
        
print("The sum of all of the numbers in L2 that are greater than 500 is:", sum_greater_500, "\n")

# 3.1) Implement the function pow(x, n), which calculates x raised to the power n (x^n). Please don't use x**n. (20pts)
def pow(x,n):
    result=1
    if n==0:
         return result
    if n<0:
         x = 1 / x
         n = -n
         
    for counter in range(n):
        result *= x
    return result

# 3.2) Calculate pow(2, 10) and pow(3, -3). (10 pts)
print("pow(2, 10) is: ", pow(2, 10))
print("pow(3, -3) is: ", pow(3, -3)) 



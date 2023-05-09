#!/usr/bin/env python
# coding: utf-8

# Q1 WAP to print all the integers that aren't divisible by either 2 or 3 and lie between 1 and 50
# 

# In[1]:


for i in range (0,51):
    if i % 2 != 0 and i % 3 !=0:
        print(i)


# Q2 Write a Python program to check the validity of a password given by the user. The password should satisfy the following criteria: 1. Contains at least one letter between a and z. 2. Contains at least one number between 1 and 9 3. Contains at least one letter between A and Z 4. Contains at least one special character from @, $, *      

# In[10]:


import re

password= input("Enter password: ")

if re.search("[a-z]", password) and re.search("[A-Z]", password) and re.search("[1-9]", password) and re.search("[@$*]", password):
    print("VALID PASSWORD")
else:
    print("INVALID PASSWORD")
    


# Q3 Write a python program to print factorial of a number. Take input from the user. 

# In[18]:


import numpy as np
import math
num = int(input("Enter a number:"))
if num<0:
    print("Cannot find a factorial")
elif num == 0:
    print("Factorial=1")
else:
    print("Factorial=",math.factorial(num))


# Q4 Write a Python program that accepts a list of numbers. Count the negative numbers and compute the sum of the positive numbers of the said list. Return these values through a list. (For example:  if input is [1, 2, 3, -4, -5] then output should be: Number of negative of numbers and sum of the positive numbers of the said list: [2, 6])

# In[110]:


numbers=input("Enter list of number, separated by space: ").split()
numbers=[int(num) for num in numbers]

neg_count=0
pos_sum=0

for num in numbers:
    if num < 0:
        neg_count =neg_count+1
    else:
        pos_sum = pos_sum+num
    
result = [neg_count, pos_sum]

print("Number of negative numbers and sum of positive numbers:", result)


# Q5 Write the python statement for the following question on the basis of given dataset: 
#     
# Name-RAM,EKLAVAY,HERRY,MARIYAM,LINDA,BILL
# Degree-BCA,MCA,BCA,BTECH,MCA,BCA
# Score-90,60,NaN,70,80,75
# 
# a) To create the above DataFrame. 
# b) To print the Degree and maximum marks in each stream. 
# c) To fill the NaN with 76. 
# d) To count the number of students in MCA. 
# e) To print the MEAN marks  for BCA

# In[113]:


import numpy as np
import pandas as pd

#a

data = {'Name': ['RAM', 'EKLAVAY', 'HERRY', 'MARIYAM', 'LINDA', 'BILL'],
        'Degree': ['BCA', 'MCA', 'BCA', 'BTECH', 'MCA', 'BCA'],
        'Score': [90, 60, np.nan, 70, 80, 75]}

df = pd.DataFrame(data)
print(df)


# In[114]:


#b
max_scores = df.groupby('Degree')['Score'].max()
print(max_scores)


# In[115]:


#c
df['Score'] = df['Score'].fillna(76)
print(df)


# In[118]:


#d
num_students_mca = df[df['Degree'] == 'MCA'].shape[0]
print(num_students_mca)


# In[119]:


#e
mean_marks_bca = df[df['Degree'] == 'BCA']['Score'].mean()
print(mean_marks_bca)


# Q6 Following is a dataset provided by a car manufacturing company
# 
# Sales- Yes,No,No,Yes,No,Yes,No,Yes,Yes
# Price- Expensive,Affordable,Expensive,Affordable,Affordable,Inexpensive,Expensive,Inexpensive,Inexpensive
# Advertising- High,Low,High,High,Low,High,High,High,Low
# Model- Sedan,Sedan,SUV,Hatchback,SUV,Hatchback,SUV,Hatchback,Hatchback
# Fuel- Electric,CNG,Petrol,CNG,Petrol,Petrol,Electric,Petrol,Electric
# 
# Fit an appropriate model on the given data to predict whether the sales would be or not. 
# Also, check the accuracy score of the model. 
# Additionally, the company is planning to launch a new electric SUV car, which will be expensive and hence company is 
# planning to do high advertising for the same.  Predict what will be the situation of sales for the same.

# In[131]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np


# In[132]:


data = {'Sales': ['Yes', 'No', 'No', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes'],'Price': ['Expensive', 'Affordable', 'Expensive', 'Affordable', 'Affordable', 'Inexpensive', 'Expensive', 'Inexpensive', 'Inexpensive'],
    'Advertising': ['High', 'Low', 'High', 'High', 'Low', 'High', 'High', 'High', 'Low'],
    'Model': ['Sedan', 'Sedan', 'SUV', 'Hatchback', 'SUV', 'Hatchback', 'SUV', 'Hatchback', 'Hatchback'],
    'Fuel': ['Electric', 'CNG', 'Petrol', 'CNG', 'Petrol', 'Petrol', 'Electric', 'Petrol', 'Electric']}
df = pd.DataFrame(data)


# In[133]:


df = pd.get_dummies(df, columns=['Sales','Price', 'Advertising', 'Model', 'Fuel'])


# In[134]:


X = df.drop('Sales_Yes', axis=1)
y = df['Sales_Yes']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[135]:


lr = LogisticRegression()
lr.fit(X_train, y_train)


# In[136]:


y_pred = lr.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


# In[137]:


print('Accuracy score:' ,accuracy)


# In[ ]:


Missing part


# Q7 The following data set is for 10 random songs. Divide the data into 3 clusters and find the accuracy of the model. (Tempo: Beats per minute of a song)
# Tempo- 101,111,67,120,107,140,156,163,103,128
# Duration- 210,285,217,152,200,123,283,206,195,208
# 

# In[66]:


import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

data={"Tempo": [101,111,67,120,107,140,156,163,103,128], "Duration": [210,285,217,152,200,123,283,206,195,208]}
df=pd.DataFrame(data)


# In[74]:


model = KMeans(n_clusters=3, random_state=42)
model.fit(df)

xyz = model.predict(df)


# In[75]:


accuracy= accuracy_score([0,1,2,0,0,2,1,0,0,0], xyz)
print("Accuracy score:", accuracy)


# Q8 Following is the data set of a company that produces automobile.
# Year':2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020,
#         'Marketing_Expense': 5, 6, 4, 8, 13, 7, 9, 10, 9.5, 11,
#         'Sales': 3.5, 4, 3, 5, 9.2, 5, 6, 7.2, 6.9, 8.5
# 
# As a marketing head of this company build a suitable regression model using the python libraries to predict the sales of company in future years. Also, check the accuracy score of the model.                               

# In[142]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split

data = {'Year': [2011, 2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020],
        'Marketing_Expense': [5, 6, 4, 8, 13, 7, 9, 10, 9.5, 11],
        'Sales': [3.5, 4, 3, 5, 9.2, 5, 6, 7.2, 6.9, 8.5]}
df = pd.DataFrame(data)


# In[143]:


X= df[["Marketing_Expense"]]
Y= df[["Sales"]]
X_train, X_test, Y_train ,Y_test= train_test_split(X,Y,test_size=0.2,random_state=42)


# In[144]:


model=LinearRegression()
model.fit(X_train,Y_train)


# In[145]:


y_pred= model.predict(X_test)


# In[146]:


accuracy= r2_score(Y_test,y_pred)
print(accuracy)


# Q9 Following is the dataset given by a school of a random sample of 10 students.
# Fit logistic regression model on the above data set to predict whether a student will take PCM or not. Also, Harsh, a male student has scored 90% on aggregate with 95% in maths and 80% in science. Find whether he would take PCM or not using the fitted regression model. (Is_PCM: Whether the student took PCM in 11th. 1 means yes. Gender: 1 is male)

# In[103]:


import pandas as pd
from sklearn.linear_model import LogisticRegression


# In[105]:


data = pd.DataFrame({
    'IS_PCM': [1, 1, 1, 0, 0, 0, 0, 1, 0, 1],
    'Aggregate Marks': [98, 97.8, 95.6, 89, 85, 98.9, 99, 95, 87.8, 94],
    'Science Marks': [97, 99, 94, 87, 86, 97, 99, 90, 94, 90],
    'Maths Marks': [98, 95, 98, 94, 84, 99, 100, 95, 80, 98],
    'Gender': [1, 1, 0, 1, 1, 0, 0, 1, 1, 1]})
df = pd.DataFrame(data)


# In[106]:


x=df[["Aggregate Marks","Science Marks","Maths Marks","Gender"]]
x.columns = ["Aggregate Marks","Science Marks","Maths Marks","Gender"]
y=df["IS_PCM"]


# In[107]:


clf = LogisticRegression().fit(x,y)


# In[108]:


harsh=[[90,80,95,1]]
harsh_df = pd.DataFrame(harsh, columns=x.columns)
pcm_prediction = clf.predict(harsh_df)


# In[109]:


if pcm_prediction ==1:
    print("Harsh is predicted to take PCM")
else:
    print("Harsh is predicted to not take PCM")


# Q10 Following is the dataset regarding the survival of passengers in a train accident
# 'Class': ['1st', '1st', '1st', '2nd', '2nd', '2nd', '2nd', '3rd', '3rd', '3rd'],
# 'Age': [29, 30, 47, 32, 57, 18, 36, 25, 18, 38],
# 'Gender': ['female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'female'],
# 'Survived': [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]
#     
# Fit an appropriate Machine learning model to predict whether the person has survived or not. 
# Also check the accuracy of the model by using the following dataset
# 
# Class	Age 	 Gender	Survived
# 1st 	13  	   male  	 0
# 2nd 	33  	  female	 1
# 3rd 	26  	   male 	 0
# 

# In[96]:


import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder


# In[97]:


train = pd.DataFrame({
    'Class': ['1st', '1st', '1st', '2nd', '2nd', '2nd', '2nd', '3rd', '3rd', '3rd'],
    'Age': [29, 30, 47, 32, 57, 18, 36, 25, 18, 38],
    'Gender': ['female', 'male', 'male', 'female', 'male', 'male', 'female', 'male', 'female', 'female'],
    'Survived': [1, 0, 1, 1, 0, 0, 1, 0, 0, 1]})


# In[98]:


test = pd.DataFrame({
    'Class': ['1st', '2nd', '3rd'],
    'Age': [13, 33, 26],
    'Gender': ['male', 'female', 'male'],
    'Survived': [0, 1, 0]})


# In[99]:


train_encoded = pd.get_dummies(train, columns=['Class', 'Gender'])
test_encoded = pd.get_dummies(test, columns=['Class', 'Gender'])


# In[100]:


X_train = train_encoded.drop(['Survived'], axis=1)
y_train = train_encoded['Survived']
model = LogisticRegression()
model.fit(X_train, y_train)


# In[101]:


X_test = test_encoded.drop(['Survived'], axis=1)
y_test = test_encoded['Survived']
y_pred = model.predict(X_test)


# In[102]:


accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)


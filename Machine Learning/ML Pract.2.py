#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np


# In[2]:


# Creating the DataFrame
data = {
    "Name": ["Alice", "Bob", "Charlie", "David", "Eve"],
    "Age": [25, 30, 22, 29, 27],
    "Score": [85, 90, 78, 88, 92]
}


# In[3]:


df = pd.DataFrame(data)


# In[4]:


# Displaying the DataFrame
print("DataFrame:\n", df)


# In[5]:


# Retrieving a single column (Age)
age_column = df["Age"]
print("\nRetrieved Column - Age:\n", age_column)


# In[6]:


# Getting summary statistics
summary = df.describe()
print("\nSummary Statistics of DataFrame:\n", summary)


# In[7]:


# Calculating mean and standard deviation for numeric columns
mean_values = df.select_dtypes(include=np.number).mean()
std_values = df.select_dtypes(include=np.number).std()


# In[8]:


print("\nMean Values:\n", mean_values)
print("\nStandard Deviation Values:\n", std_values)


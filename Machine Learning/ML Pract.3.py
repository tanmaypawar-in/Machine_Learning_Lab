#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
file_path = "Salary_dataset.csv"
df = pd.read_csv(file_path)
print(df.head())


# In[2]:


print(df.info())
print(df.describe())


# In[4]:


import numpy as np
X = df[['YearsExperience']].values
y = df['Salary'].values
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
print(f"Intercept: {model.intercept_}")
print(f"Coefficient: {model.coef_[0]}")


# In[6]:


y_pred = model.predict(X_test)
comparison = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})
print(comparison.head())


# In[7]:


from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {mse}")


# In[ ]:





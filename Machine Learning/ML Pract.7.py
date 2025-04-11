#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


# In[2]:


from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target


# In[3]:


df = pd.DataFrame(X, columns=iris.feature_names)
df['species'] = y
print(df.head())


# In[4]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# In[7]:


knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)


# In[8]:


y_pred = knn.predict(X_test)


# In[9]:


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy * 100:.2f}%")
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))


# In[10]:


for i in range(len(y_test)):
    correct = y_test[i] == y_pred[i]
    print(f"Actual: {iris.target_names[y_test[i]]}, Predicted: {iris.target_names[y_pred[i]]} - {'Correct' if correct else 'Wrong'}")


# In[ ]:





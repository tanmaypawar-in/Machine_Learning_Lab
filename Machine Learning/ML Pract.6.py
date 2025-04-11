#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn import datasets


# In[2]:


# Load Iris dataset
iris = datasets.load_iris()
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['label'] = iris.target


# In[3]:


# Splitting features and labels
X = data.iloc[:, :-1].values
y = data.iloc[:, -1].values


# In[4]:


# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[5]:


# Train Naïve Bayes classifier
model = GaussianNB()
model.fit(X_train, y_train)


# In[6]:


y_pred = model.predict(X_test)


# In[7]:


# Evaluate model
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy of Naïve Bayes classifier:", accuracy)


# In[8]:


# Display confusion matrix and classification report
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred, target_names=iris.target_names))


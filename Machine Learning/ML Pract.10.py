#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# In[3]:


def sigmoid_derivative(x):
    return x * (1 - x)


# In[4]:


def binary_cross_entropy(y_true, y_pred):
    return -np.mean(y_true * np.log(y_pred + 1e-7) + (1 - y_true) * np.log(1 - y_pred + 1e-7))


# In[7]:


def train_neural_network(X, y, epochs=10000, lr=0.1): 
 input_dim = X.shape[1]
 weights = np.random.uniform(size=(input_dim, 1))
 bias = np.random.uniform(size=(1,))

 for epoch in range(epochs):
     linear_output = np.dot(X, weights) + bias
     predictions = sigmoid(linear_output)

     loss = binary_cross_entropy(y, predictions)

     error = predictions - y
     d_pred = error * sigmoid_derivative(predictions)

     weights -= lr * np.dot(X.T, d_pred)
     bias -= lr * np.sum(d_pred)

     if epoch % 1000 == 0:
         print(f"Epoch {epoch}, Loss: {loss:.4f}")
 
 return weights, bias


# In[8]:


def predict(X, weights, bias):
    return sigmoid(np.dot(X, weights) + bias) >= 0.5


# In[9]:


X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

logic_gates = {
    "AND": np.array([[0], [0], [0], [1]]),
    "OR": np.array([[0], [1], [1], [1]]),
    "NAND": np.array([[1], [1], [1], [0]]),
    "NOR": np.array([[1], [0], [0], [0]]),
    "XOR": np.array([[0], [1], [1], [0]])
}

for gate_name, y in logic_gates.items():
    print(f"\nTraining for {gate_name} gate:")
    weights, bias = train_neural_network(X, y, epochs=10000, lr=0.1)
    predictions = predict(X, weights, bias).astype(int)
    print(f"Predictions for {gate_name} gate:\n{predictions.reshape(-1)}")


# In[ ]:





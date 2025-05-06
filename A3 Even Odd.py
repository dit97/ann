#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
class Perceptron:
    def __init__(self, input_size, learning_rate=0.1):
        self.weights = np.zeros(input_size)
        self.learning_rate = learning_rate

    def predict(self, inputs):
        return 1 if np.dot(inputs, self.weights) >= 0 else 0

    def train(self, inputs, targets, epochs=100):
        for _ in range(epochs):
            for input_vector, target in zip(inputs, targets):
                self.weights += self.learning_rate * (target - self.predict(input_vector)) * input_vector

# Convert digit to ASCII binary form
def number_to_binary(number):
    return [int(b) for b in format(ord(str(number)), '07b')]

# Input: Numbers 0-9 and output: Even -> 1, Odd -> 0
inputs = np.array([number_to_binary(i) for i in range(10)])
outputs = np.array([1 if i % 2 == 0 else 0 for i in range(10)])

# Train Perceptron
perceptron = Perceptron(7)
perceptron.train(inputs, outputs)

# Test Perceptron
for i in range(10):
    binary_input = number_to_binary(i)
    prediction = perceptron.predict(binary_input)
    print(f"Input: {i} - Predicted: {'Even' if prediction == 1 else 'Odd'}")


# In[ ]:





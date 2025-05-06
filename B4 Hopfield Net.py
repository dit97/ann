#!/usr/bin/env python
# coding: utf-8

# In[13]:


import numpy as np

class HopfieldNetwork:
    def __init__(self, size):
        self.size = size
        self.weights = np.zeros((size, size))
        
    # creating weights by hebbs rule
    def train(self, patterns):
        for p in patterns:
            p = np.reshape(p, (self.size, 1))
            self.weights += np.dot(p, p.T)  # Multiplication of p and its transpose to get weights 
        np.fill_diagonal(self.weights, 0)  # No self-connection
        self.weights /= len(patterns)      # Normalizing

    
    def recall(self, input_pattern, steps=5):
        pattern = input_pattern.copy()
        for _ in range(steps):
            for i in range(self.size):
                net_input = np.dot(self.weights[i], pattern)   # Pattern is multiplied to each weights to get pattern
                pattern[i] = 1 if net_input >= 0 else -1
        return pattern

patterns = np.array([
    [1, -1, 1, -1],
    [-1, 1, -1, 1],
    [1, 1, -1, -1],
    [-1, -1, 1, 1]
])

# Create and train the Hopfield network
hopfield_net = HopfieldNetwork(size=4)
hopfield_net.train(patterns)

# Test recall with a noisy version of a pattern
test_pattern = np.array([-1, -1, 1, -1])  # Noisy version of pattern 1
output = hopfield_net.recall(test_pattern)

print("Input Pattern: ", test_pattern)
print("Recalled Pattern:", output)


# In[ ]:





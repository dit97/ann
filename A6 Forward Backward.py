#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

class ANN:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.1):
        
        self.lr = learning_rate
        self.input_size = input_size
        self.W1 = np.random.rand(input_size, hidden_size)
        self.b1 = np.zeros((1, hidden_size))
        self.W2 = np.random.rand(hidden_size, output_size)
        self.b2 = np.zeros((1, output_size))
        
    def forward(self, X):
        self.z1 = np.dot(X, self.W1) + self.b1
        self.a1 = sigmoid(self.z1)
        
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        self.a2 = sigmoid(self.z2)
        
        return self.a2
    
    def backward(self, X, y, output):
        output_error = y - output
        output_delta = output_error * sigmoid_derivative(output)
        
        hidden_error = output_delta.dot(self.W2.T)
        hidden_delta = hidden_error * sigmoid_derivative(self.a1)
        
        self.W2 += self.a1.T.dot(output_delta) * self.lr
        self.b2 += np.sum(output_delta, axis=0, keepdims=True)  * self.lr
        self.W1 += X.T.dot(hidden_delta) * self.lr
        self.b1 += np.sum(hidden_delta, axis=0, keepdims=True) * self.lr
    
    def train(self, X, y, epochs=1000):
        for epoch in range(epochs):
            output = self.forward(X)
            self.backward(X, y, output)
            if epoch % 100 == 0:
                loss = np.mean((y - output) ** 2)
                print(f"Epoch {epoch} Loss : {loss:.4f}")
                
    def predict(self, X):
        return self.forward(X)
    
def main():
    
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    y = np.array([
        [0],
        [1],
        [1],
        [0]
    ])
    
    ann = ANN(input_size=2, hidden_size=4, output_size=1, learning_rate=0.5)
    ann.train(X, y, epochs=1000)
    
    print('\nFinal Predictions:')
    for x in X:
        output = ann.predict(x.reshape(1, -1))
        print(f'Input: {x} -> Output: {output.round(0)}')

if __name__ == '__main__':
    main()
  


# In[ ]:





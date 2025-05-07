


import numpy as np
import matplotlib.pyplot as plt

def linear_activation(x):
    return x

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def tanh_activation(x):
    return np.tanh(x)

def relu_activation(x):
    return np.maximum(0, x)

# Generate input values
x_values = np.linspace(-6, 6, 50)
print(x_values.dtype)

# Calculate activation function values
linear_values = linear_activation(x_values)
sigmoid_values = sigmoid_activation(x_values)
tanh_values = tanh_activation(x_values)
relu_values = relu_activation(x_values)

# Plotting
plt.figure(figsize=(8, 6))

plt.subplot(2, 2, 1)
plt.plot(x_values, linear_values, label='Linear')
plt.title('Linear Activation Function')
plt.legend()

plt.subplot(2, 2, 2)
plt.plot(x_values, sigmoid_values, label='Sigmoid')
plt.title('Sigmoid Activation Function')
plt.legend()

plt.subplot(2, 2, 3)
plt.plot(x_values, tanh_values, label='Tanh')
plt.title('Tanh Activation Function')
plt.legend()

plt.subplot(2, 2, 4)
plt.plot(x_values, relu_values, label='ReLU')
plt.title('ReLU Activation Function')
plt.legend()

plt.tight_layout()
plt.show()


#tan= e(X)-e(-x)/e(X)+e(-x)  [-1.1]
# np.linspace(-6, 6, 50) generates 50 equally spaced numbers from -6 to 6.
# Each plt.subplot(2, 2, n) divides the plot into a 2x2 grid and selects the nth subplot.
# plt.plot(...) plots the x and y values for each function.
# plt.title(...) sets the title.
# plt.legend() shows the label on the plot.







import numpy as np


X = np.array([
    [1, -1],   # First input vector
    [1,  1]    # Second input vector
])

Y = np.array([
    [1, -1],   # Output for first input
    [-1, -1]   # Output for second input
])


# Step 2: Create Weight MatrixStart with a zero weight matrix

W = np.zeros((2, 2))

# Multiply each pair of x & y 
for i in range(2):
    W += np.outer(Y[i], X[i])

print("Weight Matrix:")
print(W)



# This sign activation function turns positive numbers into 1, and negative into -1
def activate(vector):
    return np.where(vector >= 0, 1, -1)


def bam_recall(x_input, W):
    x = x_input.copy()
    while True:
        # Forward: X to Y
        y = activate(np.dot(W, x))

        # Backward: Y to X
        x_new = activate(np.dot(W.T, y))

        # If input stops changing, we have converged
        if np.array_equal(x, x_new):
            break
        x = x_new
    return x, y


# Test with first input
test_input1 = np.array([1, -1])
x1, y1 = bam_recall(test_input1, W)
print("\nTest Input 1:", test_input1)
print("Recalled Output (Y):", y1)
print("Recalled Input (X):", x1)

# Test with second input
test_input2 = np.array([1, 1])
x2, y2 = bam_recall(test_input2, W)
print("\nTest Input 2:", test_input2)
print("Recalled Output (Y):", y2)
print("Recalled Input (X):", x2)

#BAM (Bidirectional Associative Memory) is a hetero-associative, 
#two-layer recurrent neural network proposed by Bart Kosko in 1988.
# Uses Hebb learning rule: neuron fire together strethens





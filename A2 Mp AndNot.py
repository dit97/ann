


def activate(inputs, weights, threshold):
    # Calculate weighted sum
    weighted_sum = sum(x * w for x, w in zip(inputs, weights))
    # Apply step activation
    return 1 if weighted_sum >= threshold else 0

# Define correct weights and threshold for ANDNOT
weights = [1, -1]
threshold = 1

# Test ANDNOT logic
def test_ANDNOT():
    print("Truth Table for ANDNOT Function:")
    print("X1\tX2\tOutput")
    for x1 in [0, 1]:
        for x2 in [0, 1]:
            output = activate([x1, x2], weights, threshold)
            print(f"{x1}\t{x2}\t{output}")

test_ANDNOT()


# Feature	Description  McCulloch-Pitts model, proposed in 1943 by Warren McCulloch and Walter Pitts
# Inputs	Binary (0 or 1)
# Weights	Fixed, often all 1s ->Summation 
# Threshold	Manually set
# Output	Binary (0 or 1)
# Activation	Step function





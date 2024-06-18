import numpy as np

def relu(inputs):
    return np.maximum(0, inputs)

inputs = [1.2, 5.1]
weights = [3.1, 2.1]
bias = 3.0

output = relu(inputs[0]*weights[0] + inputs[1]*weights[1] + bias)
print(output)

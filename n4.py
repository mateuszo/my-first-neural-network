import numpy as np

def relu(inputs):
    return np.maximum(0, inputs)

inputs = [1.2, 5.1]
weights = [3.1, 2.1]
bias = 3.0

output = relu(inputs[0]*weights[0] + inputs[1]*weights[1] + bias)
print(output)


## Updated Layer class

def gen_weights(n_inputs, n_neurons):
    return 0.10 * np.random.randn(n_neurons, n_inputs)


def gen_biases(n_neurons):
    return 0.10 * np.random.randn(n_neurons)

class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = gen_weights(n_inputs, n_neurons)
        self.biases = gen_biases(n_neurons)

    def output(self, inputs):
        return relu(np.dot(self.weights, inputs) + self.biases)
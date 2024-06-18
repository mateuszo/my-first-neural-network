
import numpy as np 


## First layer
inputs = [1.0, 2.0, 3.0, 2.5]
weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2.0, 3.0, 0.5]

output = np.dot(weights, inputs) + biases
print(output)


## Second layer
weights2 = [[1, 2, 3],
            [3, 2, 1]]

biases2 = [1, 2]

output2 = np.dot(weights2, output) + biases2

print(output2)


## Layer class

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
        return np.dot(self.weights, inputs) + self.biases


# Create two neural network layers
layer1 = Layer(4, 3)
layer2 = Layer(3, 2)

l1_output1 = layer1.output(inputs)
l2_output2 = layer2.output(l1_output1)
print(l2_output2)
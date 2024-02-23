"""
Neural network that identifies whether x's or o's won the tic tac toe game.
"""

import copy
import numpy as np

np.set_printoptions(suppress=True)

from data import generate_results

"""
Data transformation logic
"""


def board_to_numerical(board):
    """
    Converts a tic-tac-toe board to a numerical representation.
    """
    flat_board = np.array(board).flatten()
    to_numerical = {"X": 1, "O": 2, " ": 0}
    return [to_numerical[x] for x in flat_board]


def winner_to_target(winner):
    """
    Converts the winner symbol to a target vector.
    """
    targets = {" ": [1, 0, 0], "X": [0, 1, 0], "O": [0, 0, 1]}
    return targets[winner]


def games_to_numerical(games):
    """
    Convert a list of games to numerical representation.
    """
    games_numerical = []

    for game in games:
        board_numerical = board_to_numerical(game["board"])
        winner_target = winner_to_target(game["winner"])
        games_numerical.append({"board": board_numerical, "winner": winner_target})

    return games_numerical


"""
Neural network logic
"""

LEARNING_RATE = 0.1


def gen_weights(n_inputs, n_neurons):
    return 0.10 * np.random.randn(n_neurons, n_inputs)


def gen_biases(n_neurons):
    return 0.10 * np.random.randn(n_neurons)


def relu(inputs):
    return np.maximum(0, inputs)


def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs))
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


def loss(inputs, target):
    inputs_clipped = np.clip(inputs, 1e-7, 1 - 1e-7)
    return -np.dot(np.log(inputs_clipped), target)


class Layer:
    def __init__(self, n_inputs, n_neurons):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = gen_weights(n_inputs, n_neurons)
        self.biases = gen_biases(n_neurons)

    def output(self, inputs):
        return relu(np.dot(self.weights, inputs) + self.biases)


# Generate 10 random games and convert them to numerical representation
games_numerical = games_to_numerical(generate_results(1000))

# Create two neural network layers
layer1 = Layer(9, 16)
layer2 = Layer(16, 3)

# Train the neural network
lowest_loss = 999
correct_count = 0
best_output = []
best_game = {}
for epoch in range(10000):

    # Adjust weights and biases
    layer1.weights += LEARNING_RATE * gen_weights(layer1.n_inputs, layer1.n_neurons)
    layer1.biases += LEARNING_RATE * gen_biases(layer1.n_neurons)
    layer2.weights += LEARNING_RATE * gen_weights(layer2.n_inputs, layer2.n_neurons)
    layer2.biases += LEARNING_RATE * gen_biases(layer2.n_neurons)

    losses = []

    for game in games_numerical:
        board = game["board"]
        winner = game["winner"]

        # Forward pass
        l1_output1 = layer1.output(board)
        l2_output2 = layer2.output(l1_output1)
        network_output = softmax(l2_output2)

        # Calculate loss
        current_loss = loss(network_output, winner)
        losses.append(current_loss)

        # Count correct predictions
        correct_count += np.argmax(network_output) == np.argmax(winner)

    # Keep track of the best parameters
    mean_loss = np.mean(losses)
    if mean_loss < lowest_loss:
        lowest_loss = mean_loss
        best_output = network_output
        best_game = game
        best_layer1 = copy.deepcopy(layer1)
        best_layer2 = copy.deepcopy(layer2)
    else:
        layer1 = copy.deepcopy(best_layer1)
        layer2 = copy.deepcopy(best_layer2)

    # Show accuracy for this epoch
    if epoch % 100 == 0:
        print(
            f"Epoch: {epoch} , Acc: {np.round((correct_count / len(games_numerical)) * 100, 2)}%"
        )
    correct_count = 0


# Print the results
print("----- Train -----")
print(f"Lowest loss: {lowest_loss}")
print(f"Best output: {best_output}")
print(f"Best game: {best_game}")
print(f"Correct: {np.argmax(best_output) == np.argmax(best_game['winner'])}")

# Test the neural network
test_game = games_numerical[0]
test_board = test_game["board"]
test_winner = test_game["winner"]


def test_network(board, winner, layer1, layer2):
    l1_output1 = layer1.output(board)
    l2_output2 = layer2.output(l1_output1)
    network_output = softmax(l2_output2)
    return network_output


test_output = test_network(test_board, test_winner, layer1, layer2)


def print_board(board):
    """
    Prints the tic-tac-toe board in a human-readable way.
    """
    to_symbol = {1: "X", 2: "O", 0: " "}
    for i in range(0, 9, 3):
        row = board[i : i + 3]
        print(f"{to_symbol[row[0]]} | {to_symbol[row[1]]} | {to_symbol[row[2]]}")
        if i < 6:
            print("-" * 9)


print("----- Test -----")
print_board(test_board)
print(f"Test winner: {test_winner}")
print(f"Test output: {test_output}")
print(f"Correct: {np.argmax(test_output) == np.argmax(test_winner)}")


"""
TODO: 
[ ] Measure accuracy
[ ] Calculate loss for each epoch
[ ] Make better test
[ ] Implement backpropagation
"""

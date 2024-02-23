import numpy as np

from data import generate_results

"""
Neural network that identifies whether x's or o's won the tic tac toe game.
"""


games = generate_results(10)


def print_game(game):
    for row in game["board"]:
        print(" ".join(row))
    print("winner:", game["winner"])


# tic tac toe board
tic_tac_toe = games[0]["board"]


# transform the board to a flat numerical array
def board_to_numerical(board):
    flat_board = np.array(board).flatten()
    to_numerical = {"X": 1, "O": 2, " ": 0}
    return [to_numerical[x] for x in flat_board]


# possible output targets
def winner_to_target(winner):
    targets = {" ": [1, 0, 0], "X": [0, 1, 0], "O": [0, 0, 1]}
    return targets[winner]


games_numerical = []

for game in games:
    board_numerical = board_to_numerical(game["board"])
    winner_target = winner_to_target(game["winner"])
    games_numerical.append({"board": board_numerical, "winner": winner_target})

print(games_numerical)


def gen_weights(n_inputs, n_neurons):
    return 0.10 * np.random.randn(n_neurons, n_inputs)


def gen_biases(n_neurons):
    return np.zeros(n_neurons)


def forward(inputs, weights, biases):
    return np.dot(weights, inputs) + biases


def relu(inputs):
    return np.maximum(0, inputs)


def softmax(inputs):
    exp_values = np.exp(inputs - np.max(inputs))
    probabilities = exp_values / np.sum(exp_values)
    return probabilities


def gen_layer(n_inputs, n_neurons):
    return {
        "weights": gen_weights(n_inputs, n_neurons),
        "biases": gen_biases(n_neurons),
    }


def loss(inputs, target):
    inputs_clipped = np.clip(inputs, 1e-7, 1 - 1e-7)
    return -np.dot(np.log(inputs_clipped), target)


layer1 = gen_layer(9, 4)
layer2 = gen_layer(4, 3)

lowest_loss = 999
best_output = []
best_game = {}

for epoch in range(1000):
    for i, game in enumerate(games_numerical):
        inputs = game["board"]
        winner = game["winner"]

        print("---- epoch:", epoch, "----")
        layer1["weights"] += 0.05 * gen_weights(9, 4)
        layer1["biases"] += 0.05 * gen_biases(4)
        layer2["weights"] += 0.05 * gen_weights(4, 3)
        layer2["biases"] += 0.05 * gen_biases(3)
        layer1_outputs = relu(forward(inputs, layer1["weights"], layer1["biases"]))
        layer2_outputs = relu(
            forward(layer1_outputs, layer2["weights"], layer2["biases"])
        )

        output = softmax(layer2_outputs)
        print(output)

        loss_value = loss(output, winner)
        if loss_value < lowest_loss:
            lowest_loss = loss_value
            best_output = output.copy()
            best_game_index = i
            best_layer1 = layer1.copy()
            best_layer2 = layer2.copy()
        else:
            layer1 = best_layer1.copy()
            layer2 = best_layer2.copy()

        print("loss:", loss_value)

print("------------")
print(best_output)
print("lowest loss:", lowest_loss)
print_game(games[best_game_index])

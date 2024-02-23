import random


def check_winner(board):
    # Check rows, columns, and diagonals for a winner
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != " ":
            return board[i][0]  # Winner in row i
        if board[0][i] == board[1][i] == board[2][i] != " ":
            return board[0][i]  # Winner in column i
    if board[0][0] == board[1][1] == board[2][2] != " ":
        return board[0][0]  # Winner in main diagonal
    if board[0][2] == board[1][1] == board[2][0] != " ":
        return board[0][2]  # Winner in anti-diagonal

    return None  # No winner


def generate_board():
    return [[" " for _ in range(3)] for _ in range(3)]


def play_game():
    board = generate_board()
    players = ["X", "O"]
    random.shuffle(players)
    current_player = players[0]

    moves = 0
    while moves < 9:
        row = random.randint(0, 2)
        col = random.randint(0, 2)
        if board[row][col] == " ":
            board[row][col] = current_player
            winner = check_winner(board)
            if winner:
                return {"board": board, "winner": winner}
            moves += 1
            current_player = players[moves % 2]
    return {"board": board, "winner": " "}


def generate_results(n):
    return [play_game() for i in range(10)]

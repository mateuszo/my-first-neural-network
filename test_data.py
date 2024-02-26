def generate_end_game_boards():
    # List to store all end game boards along with the winner
    end_game_boards = []

    # Define a helper function to recursively generate game boards
    def generate_board(board, player):
        winner = check_winner(board)
        if winner:
            end_game_boards.append((board, winner))
            return
        if is_draw(board):
            end_game_boards.append((board, " "))
            return
        for i in range(3):
            for j in range(3):
                if board[i][j] == " ":
                    new_board = [row[:] for row in board]  # Copy the board
                    new_board[i][j] = player
                    generate_board(new_board, "X" if player == "O" else "O")

    # Check if the game is a draw
    def is_draw(board):
        for row in board:
            if " " in row:
                return False
        return True

    # Check if there is a winner
    def check_winner(board):
        for i in range(3):
            if board[i][0] == board[i][1] == board[i][2] != " ":
                return board[i][0]  # Winner in row i
            if board[0][i] == board[1][i] == board[2][i] != " ":
                return board[0][i]  # Winner in column i
        if board[0][0] == board[1][1] == board[2][2] != " ":
            return board[0][0]  # Winner in main diagonal
        if board[0][2] == board[1][1] == board[2][0] != " ":
            return board[0][2]  # Winner in anti-diagonal
        return None

    # Start with an empty board and generate all possible game boards
    empty_board = [[" " for _ in range(3)] for _ in range(3)]
    generate_board(empty_board, "X")

    return end_game_boards

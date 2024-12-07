"""
Tic Tac Toe Player
"""

import copy

X = "X"
O = "O"
EMPTY = None


def initial_state():
    """Returns starting state of the board."""
    return [[EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY],
            [EMPTY, EMPTY, EMPTY]]


def player(board):
    """
    Returns the player who has the next turn on the board.
    X starts first, alternating turns.
    """
    x_count = sum(row.count(X) for row in board)
    o_count = sum(row.count(O) for row in board)
    return X if x_count <= o_count else O


def actions(board):
    """
    Returns a set of all possible actions (i, j) available on the board.
    Each action is a tuple representing row and column indices.
    """
    return {(i, j) for i in range(3) for j in range(3) if board[i][j] == EMPTY}


def result(board, action):
    """
    Returns the board that results from making move (i, j) on the board.
    Raises an exception for invalid actions.
    """
    if action not in actions(board):
        raise ValueError(f"Invalid action: {action}")
    new_board = copy.deepcopy(board)
    new_board[action[0]][action[1]] = player(board)
    return new_board


def winner(board):
    """
    Returns the winner of the game, if there is one.
    Checks rows, columns, and diagonals for three of the same symbol.
    """
    # Check rows and columns
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] and board[i][0] is not None:
            return board[i][0]
        if board[0][i] == board[1][i] == board[2][i] and board[0][i] is not None:
            return board[0][i]

    # Check diagonals
    if board[0][0] == board[1][1] == board[2][2] and board[0][0] is not None:
        return board[0][0]
    if board[0][2] == board[1][1] == board[2][0] and board[0][2] is not None:
        return board[0][2]

    return None


def terminal(board):
    """
    Returns True if the game is over (either a winner or a full board),
    False otherwise.
    """
    return winner(board) is not None or all(cell is not None for row in board for cell in row)


def utility(board):
    """
    Returns a numeric value for the board state:
    1 if X has won, -1 if O has won, 0 otherwise.
    """
    win = winner(board)
    if win == X:
        return 1
    elif win == O:
        return -1
    return 0


def minimax(board):
    """
    Returns the optimal action for the current player on the board.
    Uses Alpha-Beta Pruning for optimization.
    """
    if terminal(board):
        return None

    current = player(board)
    if current == X:
        _, move = max_value(board, float('-inf'), float('inf'))
    else:
        _, move = min_value(board, float('-inf'), float('inf'))
    return move


def max_value(board, alpha, beta):
    """
    Maximizing step of the minimax algorithm with Alpha-Beta Pruning.
    """
    if terminal(board):
        return utility(board), None

    v = float('-inf')
    best_move = None
    for action in actions(board):
        min_val, _ = min_value(result(board, action), alpha, beta)
        if min_val > v:
            v = min_val
            best_move = action
        alpha = max(alpha, v)
        if alpha >= beta:
            break
    return v, best_move


def min_value(board, alpha, beta):
    """
    Minimizing step of the minimax algorithm with Alpha-Beta Pruning.
    """
    if terminal(board):
        return utility(board), None

    v = float('inf')
    best_move = None
    for action in actions(board):
        max_val, _ = max_value(result(board, action), alpha, beta)
        if max_val < v:
            v = max_val
            best_move = action
        beta = min(beta, v)
        if beta <= alpha:
            break
    return v, best_move

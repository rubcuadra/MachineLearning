from conectaT import getBestColToPlay
import random

width, height = 7, 6

# board = [[0 for x in range(width)]for y in range (height)]
board = [[0 for y in range(width)] for x in range(height)]


# Matrix positions
# [5][0] [5][1] [5][2] [5][3] [5][4] [5][5] [5][6]
# [4][0] [4][1] [4][2] [4][3] [4][4] [4][5] [4][6]
# [3][0] [3][1] [3][2] [3][3] [3][4] [3][5] [3][6]
# [2][0] [2][1] [2][2] [2][3] [2][4] [2][5] [2][6]
# [1][0] [1][1] [1][2] [1][3] [1][4] [1][5] [1][6]
# [0][0] [0][1] [0][2] [0][3] [0][4] [0][5] [0][6]

# This is the function used to make a move.
# The function recieves the column (that is going to be given by the intelligent algorithm), and the player number that is making the move.
# If the move can't be made, the function returns -1, else: it returns the col in which it was placed.
def place(column, player_number):
    global board, width, height
    if (column >= width or column < 0):
        return -1
    partial_height = height - 1
    while (partial_height >= 0 and board[partial_height][column] == 0):
        partial_height -= 1
    if (partial_height + 1 < height):
        board[partial_height + 1][column] = player_number
        return partial_height + 1
    return -1


# Function used to check whether game has finished or not
# Return values:
# -1 -> game tie
# 0  -> game continues
# 1  -> game won by player_number

# Possible t's:

#   010     111     10      01      100     001     101     101
#   111     010     11      11      010     010     010     010
#                   10      01      101     101     001     100

def gameFinished(player_number):
    global board, width, height
    available_moves = False
    for pos in range(width):
        if (board[height - 1][pos] == 0):
            available_moves = True
            break

    if (checkAnyT(player_number)):
        return 1

    if (not available_moves):
        return -1

    return 0


# Function for printing the game board
def printGame():
    global board, width, height
    for row in range(height - 1, -1, -1):
        for col in range(0, width):
            # if(board[x][y] == 0):
            #   print (" ")
            print (board[row][col], end=" ")
        print ("\n")
    print ("\n")


def checkAnyT(player_number):
    global board, width, height
    for r in range(0, height):
        for c in range(0, width):
            if (board[r][c] == player_number):
                if (checkWinBelow(r, c, player_number)
                    or checkWinAbove(r, c, player_number)
                    or checkLeft(r, c, player_number)
                    or checkRight(r, c, player_number)
                    or checkWinBottomRight(r, c, player_number)
                    or checkWinBottomLeft(r, c, player_number)
                    or checkWinTopLeft(r, c, player_number)
                    or checkWinTopRight(r, c, player_number)):
                    return True
    return False


def checkWinBelow(row, col, player_number):
    global board, width, height
    if (col + 1 == width or col == 0 or row + 1 == height): return False
    if (board[row + 1][col - 1] == board[row + 1][col] == board[row + 1][
            col + 1] == player_number): return True
    return False


def checkWinAbove(row, col, player_number):
    global board, width, height
    if (col == 0 or row == 0 or col + 1 == width): return False
    if (board[row - 1][col - 1] == board[row - 1][col] == board[row - 1][
            col + 1] == player_number): return True
    return False


def checkLeft(row, col, player_number):
    global board, width, height
    if (row + 1 >= height or col + 1 >= width or row - 1 < 0): return False
    if (board[row - 1][col + 1] == board[row][col + 1] == board[row + 1][col + 1] == player_number): return True
    return False


def checkRight(row, col, player_number):
    global board, width, height
    if (col == 0 or row == 0 or row + 1 == height): return False
    if (board[row - 1][col - 1] == board[row][col - 1] == board[row + 1][col - 1] == player_number): return True
    return False


def checkWinBottomRight(row, col, player_number):
    global board, width, height
    if (row + 2 >= height or col - 2 < 0): return False
    if (board[row][col -2] == player_number and board[row + 1][col - 1] == player_number and board[row + 2][
            col] == player_number): return True
    return False


def checkWinBottomLeft(row, col, player_number):
    global board, width, height
    if (row + 2 >= height or col + 2 >= width): return False
    if (board[row + 2][col] == player_number and board[row + 1][col + 1] == player_number and board[row][
            col + 2] == player_number): return True
    return False


def checkWinTopLeft(row, col, player_number):
    global board, width, height
    if (row - 2 < 0 or col + 2 >= width): return False
    if (board[row - 2][col] == player_number and board[row - 1][col + 1] == player_number and board[row][
            col + 2] == player_number): return True
    return False


def checkWinTopRight(row, col, player_number):
    global board, width, height
    if (row - 2 < 0 or col - 2 < 0): return False
    if (board[row - 2][col] == player_number and board[row - 1][col - 1] == player_number and board[row][
            col - 2] == player_number): return True
    return False


def intelligentFunction1(turn, board):
    return getBestColToPlay(board,turn,depth=1)

def intelligentFunction2(turn, board):
    while True:
        try:
            move = int(input('Enter your move!'))
            if move < 0 or move >= width: raise Exception("No valido")
            return move 
        except KeyboardInterrupt: raise
        except:print ("Solo se aceptan numeros de 0 - 6")

def main():
    global board
    turn = 1
    loser = 0
    while (gameFinished(turn) == 0):
        printGame()
        if (turn == 1):
            turn = 2
        else:
            turn = 1
        if (turn == 1):
            column = intelligentFunction1(turn, board)
        if (turn == 2):
            column = intelligentFunction2(turn, board)
        if (place(column, turn) == -1):
            loser = turn
            break

    # Game is a tie
    if (gameFinished(turn) == -1): print("The game is a tie!")
    elif not (loser == 0): print ("The loser is ", turn)
    else:
        printGame()
        print ("The winner is ", turn)


if __name__ == '__main__':
    main()
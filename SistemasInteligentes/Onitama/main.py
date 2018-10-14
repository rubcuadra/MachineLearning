from board import OnitamaBoard
from agent import OnitamaPlayer

if __name__ == '__main__':
	board = OnitamaBoard()
	op = OnitamaPlayer(board.BLUE,3)
	print(board)
	print(op.getBestMovement(board))
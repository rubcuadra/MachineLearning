from board import OnitamaBoard
from agent import OnitamaPlayer

if __name__ == '__main__':
    board = OnitamaBoard()
    op  = OnitamaPlayer(board.RED,1)
    op2 = OnitamaPlayer(board.BLUE,2)
    print(board)
    while not board.isGameOver():
        movement = op.getBestMovement(board)
        if board.canMove( op.player,*movement ):
            board = board.move(op.player,*movement) 
        else:
            print("La cago op")
            raise
        print("RED",movement)
        print(board)
        if board.isGameOver():  break 
        movement = op2.getBestMovement(board)
        if board.canMove( op2.player,*movement ):
            board = board.move(op2.player,*movement) 
        else:
            print("La cago op2")
            raise
        print("BLUE",movement)
        print(board)
    print(f"Winner is { 'BLUE' if board._blue_is_alive else 'RED' }")
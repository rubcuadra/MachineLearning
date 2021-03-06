from reversi_player import Agent
from reversi import ReversiBoard
from random import shuffle 

_agent2 = None
def P1Turn(board, player): #Player
    move = input('Enter your move: ')
    return move

def P2Turn(board, player): #IA
    return _agent2.getBestMovement(board, player) 

#Nivel  => Profundidad de busqueda
#Fichas => 0 Blancas, 1 Negras (Para la computadora) 
#Inicia => 0 Computadora, 1 Contrario 
def othello(nivel, fichas=1, inicio=1):
    global _agent2
    _agent2 = Agent(nivel)     #Crear agente para P2

    #P2 is the computer
    board = ReversiBoard()
    print("=== GAME STARTED ===")
    print(f"{board.P2S} = Blacks")
    print(f"{board.P1S} = Whites\n")
    print(board)
    
    #Who starts and set tokens
    if inicio == 1:
        order  = ["P1","P2"]
        turns  = [P1Turn,P2Turn]
        tokens = [board.P2,board.P1] if fichas == 0 else [board.P1,board.P2]
    else:
        order  = ["P2","P1"]
        turns  = [P2Turn,P1Turn]
        tokens = [board.P1,board.P2] if fichas == 0 else [board.P2,board.P1]
    
    while not ReversiBoard.isGameOver(board):    
        for i in range(2):
            P1Score = board.score( board.P1 )
            P2Score = board.score( board.P2 )
            print("Scores:\t",f"{board.P1S}:{P1Score}","\t",f"{board.P2S}:{P2Score}")
            if board.canPlayerMove( tokens[i] ) : 
                print(f"{order[i]} turn, throwing {board.getSymbol(tokens[i])}")
                while True:
                    move = turns[i]( ReversiBoard( board.value )  ,tokens[i])  
                    if ReversiBoard.cellExists(move):
                        r = board.throw(tokens[i],move)
                        if len(r) > 0: 
                            print(f"Selection: {move}")
                            board.doUpdate(tokens[i],r)
                            break
                    print("Wrong movement, try again")
                print(board)

    if P1Score == P2Score: print("TIE !!")
    else:                  print(f"Winner is {board.P1S if P1Score>P2Score else board.P2S}")

if __name__ == '__main__':
    level  = 2 #Dificultad de la AI
    npc    = 0 #Fichas del P2(AI). 0 es Blancas
    starts = 1 #0 => P2 (AI) empieza
    othello(level,npc,starts)
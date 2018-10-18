from board import OnitamaBoard
from movements import OnitamaCards

class BoardNode(object):
    def __init__(self, board, player, movement=None):
        super(BoardNode, self).__init__()
        self.board         = board
        self.player        = player   
        self.children      = [] 
        self.score         = self.getScore()
        if movement: self.movement = movement #Backtracking Ej. (0,3), "RABBIT", (1,2)
    
    #Call iterator with the same player or with the opponent
    def updateChildren(self, change_player = False):
        player = OnitamaBoard.getOpponent(self.player) if change_player else self.player
        self.children = [ bn for bn in self.childrenIterator(player)]

    #Get all possible children nodes given one player
    def childrenIterator(self, player): 
        cards = self.board.cards[player]
        
        if player is OnitamaBoard.RED:
            positions = self.board._red_pos  | set([self.board._red_master_pos])
            getMov = lambda x,y: (x[0] - y[0],x[1] - y[1])
        else:
            positions = self.board._blue_pos | set([self.board._blue_master_pos])
            getMov = lambda x,y: (x[0] + y[0],x[1] + y[1])

        for token in positions:                     #Por cada estatua
            for card in cards:                      #Por cada tarjeta
                for movement in OnitamaCards[card]: #Por cada celda de la tarjeta
                    toCell = getMov(token,movement)
                    if self.board.canMove(player, token, card, toCell):
                        yield BoardNode(self.board.move(player, token, card, toCell),player,movement=(token,card,toCell))
                        
    #Heuristic - utilidad, Mayor es mejor
    def getScore(self): 
        #Count es Mis tokens - Opponent tokens
        if self.player is OnitamaBoard.RED:
            finalPos   = (4,2)
            currentPos = self.board._red_master_pos   
            killed = not self.board._blue_is_alive
            count = self.board._red  - self.board._blue  
        else:
            finalPos   = (0,2)
            currentPos = self.board._blue_master_pos
            count = self.board._blue - self.board._red
            killed = not self.board._red_is_alive
        #Distancia manhattan maestro vs destino
        h = abs(finalPos[0] - currentPos[0]) + abs(finalPos[1] - currentPos[1])
        if h == 0 or killed: return 800 #Muchos puntos, ya se gano 
        #Ajustar magnitudes de impacto
        return (6-h)*20 + count*80 #Distancia + Numero de piezas

    #For Max and Min
    def __lt__(self,other):
        return self.score < other.score

    def __str__(self):
        return str(self.board)
    
    @staticmethod
    def MM(node,level=0): #Regresa scores
        if level <= 0 or node.board.isGameOver(): return node.score
        node.updateChildren(change_player=True) #Expandir y sacar Max o Min de los hijos
        if node.children: 
            if level%2 == 0: #MAX
                return max( [ BoardNode.MM(c,level-1) for c in node.children ] ) 
            else: #MIN   
                return min( [ BoardNode.MM(c,level-1) for c in node.children ] ) 
        return node.score #Se intento expandir pero ya no se puede

#Level 1 = Expande 1 tiro nuestro y 1 de oponente
class OnitamaPlayer():
    def __init__(self, player, level):
        self.level  = 1+(level-1)*2 #La primera es max sin cambiar oponente, lo demas empieza logica minmax alternando
        self.player = player
    
    def getBestMovement(self, board): 
        root = BoardNode(board,self.player)    #Root
        root.updateChildren()                  #First Layer, get Max
        
        #Get Max but save the movement
        maxMovement = {"movement":None,"score":float("-inf")}
        
        for childNode in root.children:
            score = BoardNode.MM(childNode, self.level )
            if score > maxMovement["score"]: 
                maxMovement = {"movement":childNode.movement,"score":score} 
        
        return maxMovement["movement"]

# RUN THE EXAMPLE AND IT will tell you the best movement in that case
if __name__ == '__main__':
    from sys import argv
    #Example: 
    #   Params -                       "BOARD;cards" PLAYER LEVEL
    #   python agent.py " bR   B  rrb      rrb    ;MONKEY FROG;HORSE MANTIS;EEL" B 2
    # argv[0] => thisFile.py
    board  = OnitamaBoard.fromArgs(argv[1]) 
    player = OnitamaBoard.BLUE if argv[2]=='B' else OnitamaBoard.RED 
    level  = int(argv[3]) #1-3
    op = OnitamaPlayer(player,level)
    movement = op.getBestMovement(board)
    #fromRow,fromCol;MOVEMENT;toRow,toCol
    print( f"{movement[0][0]},{movement[0][1]};{movement[1]};{movement[2][0]},{movement[2][1]}" )


from string import ascii_uppercase as UC
from collections import defaultdict
from random import shuffle

#Generate indexes and throws once
boardSize = 8
cells_ix = {}#defaultdict(lambda: (-1,-1))
ix_cells = {}#defaultdict(lambda: ' 0')
for i,l in enumerate(UC[:boardSize]):
    for j in range(boardSize): 
        cells_ix[f"{l}{j+1}"] =  (j,i)
        ix_cells[(j,i)]  = f"{l}{j+1}"

class ReversiBoard(object):
    P1,P1S       =  1, "X"       
    P2,P2S       = -1, "O"
    EMPTY,EMPTYS =  0, " "
    
    def __init__(self, value=None):
        super(ReversiBoard, self).__init__()
        h = int(boardSize/2)
        if not value:
            self.value = [ [self.EMPTY]*boardSize for _ in range(boardSize)]
            #Set Diagonal in the center
            self.value[h  ][h  ] = self.P1
            self.value[h-1][h-1] = self.P1
            self.value[h  ][h-1] = self.P2
            self.value[h-1][h  ] = self.P2
        else:
            self.value = val

    def __getitem__(self,i): return self.value[i]

    def compare(self,i,j,player):
        if i < 0 or i >= boardSize or j < 0 or j >= boardSize: return False #OutOfIndex
        return self[i][j] == player #Check
    
    def doUpdate(board,player,throwResult):
        for i,j in throwResult: board[i][j] = player

    def throw(board,player,position):    
        i,j =  board.cellToIndex(position)
        toConvert = []
        if board[i][j] is board.EMPTY: #Check if that place is empty
            opponent = board.getOpponent(player) #Check adjacent opponents discs and save them
            temp,add = [],False
            if board.compare(i-1,j,opponent): #Check Up
                for d in range(i-1,-1,-1):
                    if board[d][j] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (d,j) ) 
                        continue
                    elif board[d][j] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i+1,j,opponent) : #Check Down
                for d in range(i+1,boardSize):
                    if board[d][j] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (d,j) ) 
                        continue
                    elif board[d][j] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i,j-1,opponent): #Check Left
                for d in range(j-1,-1,-1):
                    if board[i][d] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (i,d) ) 
                        continue
                    elif board[i][d] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i,j+1,opponent): #Check Right
                for d in range(j+1,boardSize):
                    if board[i][d] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (i,d) ) 
                        continue
                    elif board[i][d] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i-1,j+1,opponent): #Check Up-Right
                e = i
                for d in range(j+1,boardSize):
                    e-=1
                    if e<0: break
                    if board[e][d] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (e,d) ) 
                        continue
                    elif board[e][d] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i-1,j-1,opponent): #Check Up-Left
                e = i
                for d in range(j-1,-1,-1):
                    e-=1
                    if e<0: break
                    if board[e][d] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (e,d) ) 
                        continue
                    elif board[e][d] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i+1,j-1,opponent): #Checar Down-Left
                e = i
                for d in range(j-1,-1,-1):
                    e+=1
                    if e>=boardSize: break
                    if board[e][d] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (e,d) ) 
                        continue
                    elif board[e][d] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if board.compare(i+1,j+1,opponent): #Checar Down-Right
                e = i
                for d in range(j+1,boardSize):
                    e+=1
                    if e>=boardSize: break
                    if board[e][d] == opponent:             #Ir agregando coordenadas a convertir
                        temp.append( (e,d) ) 
                        continue
                    elif board[e][d] == player: add = True  #Atrapado por player, salvar coordenadas
                    break 
                if add: toConvert += temp
                temp,add = [],False
            if len( toConvert ) > 0: toConvert.append( (i,j) )
        return toConvert
    
    def canP1Move(self):
        for move in cells_ix:
            if len( self.throw( self.P1, move) ) > 0: 
                return True
        return False

    def canP2Move(self):
        for move in cells_ix:
            if len( self.throw( self.P2, move) ) > 0: 
                return True
        return False

    @staticmethod
    def isGameOver(board): 
        return not (board.canP1Move() or board.canP2Move())

    def score(board, player , s=0):  #Count disks of that user (should we do it minus the ones of the other?)
        for row in board:
            for cell in row: 
                if cell == player: s+=1
        return s

    @staticmethod
    def cellToIndex(cellName):  return cells_ix[cellName] #Dict

    @staticmethod
    def indexToCell(cellIx):    return ix_cells[cellIx] #Dict

    @staticmethod
    def getOpponent(p): return ReversiBoard.P1 if p == ReversiBoard.P2 else ReversiBoard.P2

    @staticmethod
    def getSymbol(v): #For printing
        if v is ReversiBoard.P1: return ReversiBoard.P1S
        if v is ReversiBoard.P2: return ReversiBoard.P2S
        return ReversiBoard.EMPTYS

    def __str__(self):
        toRet = f"   {' '.join(UC[:boardSize])}\n" 
        for i,row in enumerate(self.value, start=1):
            toRet += f" {i}|"
            for cell in row:
                toRet += f"{ self.getSymbol(cell)}|"
            toRet += "\n"
        return toRet

def P1Turn(board, player):
    p = board.getSymbol(player)
    print(f"P1 turn, throwing {p}")
    while True:
        move = input('Enter your move: ')
        if move in cells_ix:
            r = board.throw(player,move)
            #It is a valid movement
            if len(r) > 0: return r
        print("Wrong movement, try again")

def P2Turn(board, player):
    p = board.getSymbol(player)
    print(f"P2 turn, throwing {p}")
    while True:
        move = input('Enter your move: ')
        if move in cells_ix:
            r = board.throw(player,move)
            #It is a valid movement
            if len(r) > 0: return r
        print("Wrong movement, try again")
        


if __name__ == '__main__':
    print("=== GAME STARTED ===\n")
    board = ReversiBoard()
    turns = [P1Turn,P2Turn]
    shuffle( turns ) #Shuffle starting order
    re = []
    print(board)
    while not ReversiBoard.isGameOver(board):
        P1Score = board.score( board.P1 )
        P2Score = board.score( board.P2 )
        print("Scores:\t",f"{board.P1S}:{P1Score}","\t",f"{board.P2S}:{P2Score}")
        if board.canP1Move(): 
            re = turns[0](board,board.P1)  
            board.doUpdate(board.P1,re)
            print(board)
        if board.canP2Move(): 
            re = turns[1](board,board.P2)  
            board.doUpdate(board.P2,re)
            print(board)
    

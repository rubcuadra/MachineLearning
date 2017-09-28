import numpy as np
from copy import copy
from random import random,randint
from sys import maxsize

class BoardNode():
    def __init__(self, data):
        self.data = data
        self.children = []

    def add_children_boards(self, boardsArray):
        for board in boardsArray:
            self.children.append( BoardNode(board) )

    def getMax(self): #Nos devuelve el hijo con mejor score y su index
        maxi = 0
        maxScore = Board.minScore
        for i,c in enumerate(self.children):     #Nodos Hijos
            if c.data is None: continue
            if maxScore<c.data.score:
                maxi = i
                maxScore = c.data.score
        return (maxi,maxScore)

class Board():
    minScore = -1*maxsize
    def __init__(self,nativeBoard=None):
        self.val = None
        self.score = self.minScore
        if nativeBoard:
            self.val = np.array(nativeBoard).transpose()[::-1,:] #Fix para que salga como deberia

    def __getitem__(self,i):
         return self.val[i]

    def __str__(self):
        return str(self.val)

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __copy__(self):
        newone = type(self)()
        newone.val = self.val.copy()
        return newone

    def getCombinations(self, diskToThrow): #Regresamos copias
        gc = []
        for i in range(self.val.shape[1]):            
            gc.append(Board.throw(copy(self),i,diskToThrow))
        return gc 
    
    def throw(brd,col_i,disk):
        ix = np.where(brd.val[:,col_i]==0)[0]
        if len(ix)==0:return None                    #Ya no se puede tirar ahi
        brd.val[:,col_i][ix[-1]] = disk              #Hacer el tiro
        brd.score = Board.getBoardScore(brd,disk,col_i) #Actualizar el score
        return brd                                   #Regresar tablero modificado

    def getSize(self):
        return self.val.shape

    #Regresa la celda o None
    def g(self,row,column): 
        try:    return self[row][column]
        except: return None

    def getBoardScore(_board, ourDisk, movedColumn):
        if not _board: return Board.minScore  #Regresar el minimo   
        rows,columns = _board.getSize() #Saber si ya nos pasamos de rows o columns
        #movedColumn = movedColumn      #Es el que nos pasaron
        movedRow     = _board.val[:,movedColumn].nonzero()[0][0]
        
        #Celdas para validar, pueden ser None
        current = _board[movedRow][movedColumn]
        der = _board.g(movedRow,movedColumn+1)
        izq = _board.g(movedRow,movedColumn-1)
        up = _board.g(movedRow-1,movedColumn)
        dwn = _board.g(movedRow+1,movedColumn)
        #Diagonales directas - grado 1, solo se cambia el grado
        upDer  = _board.g(movedRow-1,movedColumn+1)
        dwnDer = _board.g(movedRow+1,movedColumn+1)
        upIzq  = _board.g(movedRow-1,movedColumn-1)
        dwnIzq = _board.g(movedRow+1,movedColumn-1) 
        
        return randint(100,8000) #Un numero grande

def getBestColToPlay(nativeBoard,disk,totalColumns=7,depth=3):
    _board = Board(nativeBoard) #Crear un tablero nuestro
    
    root = BoardNode( _board )
    root.add_children_boards( _board.getCombinations(disk) )
    # for i in range(depth-1):                        #depths hacia abajo
    #     d = disk if not i%2 else 666 #cualquier otro numero
    #     calculateScores(population,d)    
    colToThrow,colScore = root.getMax()
    return colToThrow
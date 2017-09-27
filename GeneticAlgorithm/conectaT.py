import numpy as np
from copy import copy
from random import random,randint
from sys import maxsize

class Board():
    def __init__(self,nativeBoard=None):
        self.val = None
        if nativeBoard:
            self.val = np.array(nativeBoard).transpose()[::-1,:] #Fix para que salga como deberia

    def __getitem__(self,i):
         return self.val[i]

    def __str__(self):
        return str(self.val)

    def __copy__(self):
        newone = type(self)()
        newone.val = self.val.copy()
        return newone

    def getCombinations(self, diskToThrow): #Regresamos copias
        return [Board.throw(copy(self),i,diskToThrow) for i in range(self.val.shape[1])]
    
    def throw(brd,col_i,disk):
        ix = np.where(brd.val[:,col_i]==0)[0]
        if len(ix)==0:return None #Ya no se puede tirar ahi
        brd.val[:,col_i][ix[-1]] = disk
        return brd                #Regresar tablero modificado

    def getSize(self):
        return self.val.shape

    #Regresa la celda o None
    def g(self,row,column): 
        try:    return self[row][column]
        except: return None

    def getBoardScore(_board, ourDisk, movedColumn):
        if not _board:        #No hay board
            return -1*maxsize      #Regresar - demasiados puntos
        
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

def getScores(boards,disk):
    r = np.array([])
    for i,b in enumerate(boards):
        r = np.append(r, Board.getBoardScore(b,disk,i))
    return r

def getBestColToPlay(nativeBoard,disk,totalColumns=7):
    _board = Board(nativeBoard) #Crear un tablero nuestro
    population       = _board.getCombinations(disk)
    #Por cada tablero en population podemos obtener nuevas populations -> Bajar en el arbol
    populationScores = getScores(population,disk)
    #Si quisieramos a modo de arbol seria loopear para sacar el bestSpecimen
    bestSpecimen     = np.argmax(populationScores)
    return bestSpecimen
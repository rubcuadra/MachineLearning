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

    def getBoardScore(boardToTest, ourDisk):
        if not boardToTest:        #No hay board
            return -1*maxsize      #Regresar - demasiados puntos
        print(boardToTest) 
        #Heuristic
        #print (boardToTest)
        #print (boardToTest[-1][1]) #Fila de hasta abajo, columna 1
        return randint(100,8000) #Un numero grande

def getBestColToPlay(nativeBoard,disk,totalColumns=7):
    ff = np.vectorize( Board.getBoardScore ) #Evaluar cada tablero y regresar su Score                               
    _board = Board(nativeBoard) #Crear un tablero nuestro
    population       = _board.getCombinations(disk)
    #Por cada tablero en population podemos obtener nuevas populations -> Bajar en el arbol
    populationScores = ff( population,disk ) 
    #Si quisieramos a modo de arbol seria loopear para sacar el bestSpecimen
    bestSpecimen     = np.argmax(populationScores)
    return bestSpecimen
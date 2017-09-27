import numpy as np
from copy import copy
from random import random,randint

class Board():
	def __init__(self, nativeBoard):
	    self.val = np.array(nativeBoard).transpose()[::-1,:] #Fix para que salga como deberia

	def __getitem__(self,i):
         return self.val[i]

	def getCombinations(self, diskToThrow):
		cols = self.val.shape[0] # 0 -> Cols ; 1 -> Rows
		combinations = []
		for i in range(cols):
			combinations.append( Board.throw( copy(self), i, diskToThrow) )
		return combinations
	
	def throw(brd,col,disk):

		return brd

	def getBoardScore(boardToTest, ourDisk):
		#Heuristic
		print (boardToTest)
		print (boardToTest[-1][1]) #Fila de hasta abajo, columna 1
		return randint(100,8000) #Un numero grande

	def __str__(self):
		return str(self.val)

def getBestColToPlay(nativeBoard,disk,totalColumns=7):
	ff = np.vectorize( Board.getBoardScore ) #Evaluar cada tablero y regresar su Score								 
	_board = Board(nativeBoard) #Crear un tablero nuestro

	population       = _board.getCombinations(disk)
	#Por cada tablero en population podemos obtener nuevas populations -> Bajar en el arbol
	populationScores = ff( population,disk ) 
	#Si quisieramos a modo de arbol seria loopear para sacar el bestSpecimen
	bestSpecimen     = np.argmax(populationScores)
	return bestSpecimen
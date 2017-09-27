import numpy as np
from copy import copy
from random import random,randint

class Board():
	def __init__(self, nativeBoard):
	    self.val = np.array(nativeBoard)

	def getCombinations(self, diskToThrow):
		cols = self.val.shape[0] # 0 -> Cols ; 1 -> Rows
		combinations = []
		for i in range(cols):
			combinations.append( Board.throw( copy(self), i, diskToThrow) )
		return combinations
	
	def throw(brd,col,disk):
		#TODO
		return brd

	def getBoardScore(boardToTest, ourDisk):
		#Heuristic
		return randint(100,8000) #Un numero grande

def getBestColToPlay(nativeBoard,disk,totalColumns=7):
	_board = Board(nativeBoard) #Crear un tablero nuestro

	ff = np.vectorize( Board.getBoardScore ) #Evaluar cada tablero y regresar su Score        										 
	population       = _board.getCombinations(disk)
	#Por cada tablero en population podemos obtener nuevas populations -> Bajar en el arbol
	populationScores = ff( population,disk ) 
	#Si quisieramos a modo de arbol seria loopear para sacar el bestSpecimen
	bestSpecimen     = np.argmax(populationScores)
	return bestSpecimen
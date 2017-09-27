import numpy as np
from random import random,randint

class Board():
	def __init__(self, nativeBoard):
	    self.val = np.array(nativeBoard)

	def getCombinations(self, diskToThrow):
		cols = self.val.shape()
		print cols
		combinations = np.array([])
		return combinations

	def getBoardScore(boardToTest, ourDisk):
		return randint(100,8000) #Un numero grande

def getBestPlay(nativeBoard,disk,totalColumns=7):
	_board = Board(nativeBoard) #Crear un tablero nuestro

	ff = np.vectorize( getBoardScore ) #Evaluar cada tablero y regresar su Score        										 
	population       = _board.getCombinations(disk)
	#Por cada tablero en population podemos obtener nuevas populations -> Bajar en el arbol
	populationScores = ff( population,disk ) 
	#Si quisieramos a modo de arbol seria loopear para sacar el bestSpecimen
	bestSpecimen     = population[np.argmax(populationScores)]
	return bestSpecimen
#Ruben Cuadra A01019102
from random import randint,seed
from enum import Enum
seed(0)
class QueenMovements(Enum): 
    '''
        We only move them up and down because there's 
        already 1 Queen in each column of the board 
        it makes no sense to move horizontally or diagonally
    '''
    UP    = "U"
    DOWN  = "D"

class QueensBoard(object):
    """
        Class that represents the board of QxQ with 1 Queen in each column
    """
    def __init__(self, Q, randomize=False):
        super(QueensBoard, self).__init__()
        self.positions = [i for i in range(Q)] # It starts all the queens in the main diagonal
        self.Q = Q
        
        if randomize: #Random positions for the queens
            for i in range(Q):
                self.positions[i] = randint(0,Q-1)
    
    def __str__(self):
        #Columns indexes
        colIndexes = '|'.join( [str(i) for i in range(self.Q) ] )
        res = f" |{colIndexes}|\n" 
        for i in range(self.Q):     #Iterate over Rows
            res+=f"{i}|"            #Row index
            for j in range(self.Q): #Iterate over Columns
                if self.positions[j] == i:  res += "Q|"
                else:                       res += " |"
            res+="\n"
        return res

#Q Number of Queens (The board has size QxQ)
#S Bool Flag that allows side movements, If True Hill Climbing algorithm allows lateral movements (Move to a node with same score)
#T Number of tries
def busquedaHC(Q=8,S=True,T=5):
    B = QueensBoard(Q,True)
    print(B.positions)
    print(B)

if __name__ == '__main__':
    N = 8
    lateral = True
    M = 5
    busquedaHC(N, lateral, M)
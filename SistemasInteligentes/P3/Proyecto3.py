#Ruben Cuadra A01019102
from queue import PriorityQueue as PQ
from random import randint, seed
from multiprocessing import Pool
from collections import Counter
from os import cpu_count
from enum import Enum
seed(1)
class QueenMovements(Enum): 
    '''
        We only move them up and down because there's 
        already 1 Queen in each column, it makes no sense 
        to move them horizontally or diagonally
    '''
    UP    = 0
    DOWN  = 1

class QueensBoard(object):
    """
        Class that represents the board of QxQ with 1 Queen in each column
    """
    def __init__(self, Q, positions=[]):
        super(QueensBoard, self).__init__()
        self.Q = Q
        if positions: #Must be an array len = Q, vals must be numbers between 0 and Q-1
            self.positions = positions
        else:         
            self.positions = [0]*Q  #Init the array
            for i in range(Q):      #Randomize all the positions
                self.positions[i] = randint(0,Q-1)
        
        #Evaluate that board
        self.score = self.evaluate()
    
    def __hash__(self):
        return hash( tuple(self.positions) ) 

    #Cost was calculated in constructor
    def __lt__(self,other): 
        return self.score < other.score
    def __le__(self,other): 
        return self.score <= other.score

    def queenAtCell(self,row,col):
        return self.positions[col] == row

    #We can improve this by  giving the movement done and the previous value
    #Heuristic, it is the number of attacks between queens, can be > Q
    def evaluate(self):
        #Count repeated in array
        count = sum( [v-1 for v in Counter(self.positions).values()] ) 
        #Check diagonals
        for col,row in enumerate( self.positions ) :
            #Upper Left
            tr = row-1
            tc = col-1
            while tr>=0 and tc>=0:
                if self.queenAtCell(tr,tc): count+=1 ; break
                tr -= 1; tc -= 1
            #Upper Right
            tr = row-1
            tc = col+1
            while tr>=0 and tc<self.Q:
                if self.queenAtCell(tr,tc): count+=1 ; break
                tr -= 1; tc += 1
            #No need to check Lower L-R
        return count

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
    #Can the queen,in that column, do that movement?
    def canMove( self, columnIndex , movement): 
        row = self.positions[columnIndex]
        if movement is QueenMovements.UP:  return row-1 > -1 
        if movement is QueenMovements.DOWN:return row+1 < self.Q

    #Should call canMove before
    # Returns a QueensBoardObject with the Movement Done
    def _move(self, columnIndex , movement): 
        newPositions = self.positions.copy()
        if movement is QueenMovements.UP:   newPositions[columnIndex]-=1
        if movement is QueenMovements.DOWN: newPositions[columnIndex]+=1
        return QueensBoard(self.Q, newPositions)

    def getCombinations(self): #Iterator
        for m in QueenMovements:    #TRY all existing movements 
            for i in range(self.Q): #in ALL columns
                if self.canMove( i,m ):
                    yield self._move(i,m) 

def HC(Q,S=False):
    betterNeighbor  = lambda x,y: x<=y if S else x<y 
    currentB = QueensBoard(Q)
    visited = set()
    while True:   
        structure = PQ()
        #Create neighbors and add them to the priority queue
        for combination in currentB.getCombinations(): structure.put( combination  )
        
        nextB = structure.get() #Pop the best
        if S: #Can move to the side
            visited.add(currentB)
            while (nextB in visited) and (not structure.empty()): nextB = structure.get()
            if nextB in visited: break #Ya se visito todo aqui                            
        if not betterNeighbor(nextB,currentB): break
        # if nextB.score < currentB.score: print(nextB.score)
        currentB = nextB

    return currentB

#Q Number of Queens (The board has size QxQ)
#S Bool Flag that allows side movements, If True Hill Climbing algorithm allows lateral movements (Move to a node with same score)
#T Number of tries
#IF there is no solution and T is inf then it will never end...
def busquedaHC(Q=8,S=False,T=float("inf"), parallelize=False):
    #Compare using Side flag is <= , else is <
    print(f"Algoritmo 'Hill Climbing' {'con' if S else 'sin'} movimientos laterales")
    found,i = None,0

    #In Line or parallel
    if parallelize: 
        rs = 10 #10 sub procs per iteration
        roundSize = T%rs if T != float("inf") else rs #Fix si piden numeros con residuo al dividir por 10
        while not found and i<T:
            pool = Pool(cpu_count()) #Parallelize crawlers
            res = pool.starmap(HC, [(Q,S)]*roundSize ) #ir de roundSize
            pool.close()
            pool.join()
            for r in res:
                i+=1
                if r.score == 0: found = r
            roundSize = rs
    else:
        while i<T:
            answer = HC(Q,S)
            if answer.score == 0:
                found = answer
                break
            i+=1
    if found:
        print(f"\nSolucion encontrada en el intento {i}")
        print(found)
    else:
        print(f"\nSolucion no encontrada en {i} intentos")
    
if __name__ == '__main__':
    N = 10
    lateral = False
    M = 9300
    busquedaHC(N, lateral, M)
    
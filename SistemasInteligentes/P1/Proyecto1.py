#RUBEN CUADRA A01019102
from collections import deque
from enum import Enum

class Movements(Enum):
    UP    = "U"
    DOWN  = "D"
    LEFT  = "L"
    RIGHT = "R"

class PuzzleNode(object): #Wrapper for Tree functionalities
    def __init__(self, _puzzle, parentNode=None, edge=None):
        super(PuzzleNode, self).__init__()
        self.state      = _puzzle
        self.parentNode = parentNode
        self.edge       = edge if parentNode!=None else None #It is the action/value that connects with the parentNode

    def getCombinations(self): #Iterator
        for m in Movements:    #TRY all existing movements
            if self.state.canMove( m ):
                yield PuzzleNode( self.state.getMovement(m), parentNode=self, edge=m ) #Move returns a Puzzle object
        
    def backTrack(self):
        if self.parentNode is None: return []
        return self.parentNode.backTrack() + [self.edge.value] #enums have .value

class Puzzle(object):
    EMPTY_SPACE = 0
    
    def __init__(self, val, ix = None):
        super(Puzzle, self).__init__()
        #Convert from [[0, 1, 2], [4, 5, 3], [7, 8, 6]] to ((0, 1, 2), (4, 5, 3), (7, 8, 6))
        self.shape = ( len(val), len(val[0]) )
        self.board = tuple( map(tuple,val) )  #Tuples are hashables TODO - improve this structure
        self.ix    = None

        #Find the empty space
        if ix is None: #We need to find the index of the 0
            for i,row in enumerate(val):
                for j,cell in enumerate(row):
                    if cell == self.EMPTY_SPACE:
                        self.ix = (i,j)     #Save and break
                        break
                if self.ix != None:         #Alerady got it
                    break
        else:
            self.ix = ix #(row,column)
        

    def __eq__(self, other):
        return self.board == other.board

    def __hash__(self):
        return hash(self.board)

    def getMovement(self,movement): #We assume that canMove was called already 
        matrix = list( map(list, self.board) ) #Copy that can be edited but it is not hashable - TODO improve this structure
        
        if movement is Movements.UP:
            matrix[  self.ix[0]   ][ self.ix[1] ] = matrix[ self.ix[0]-1 ][ self.ix[1] ]
            matrix[  self.ix[0]-1 ][ self.ix[1] ] = self.EMPTY_SPACE
            newIx = (self.ix[0]-1 ,  self.ix[1] )
            return Puzzle( matrix, newIx )
        if movement is Movements.DOWN:
            matrix[  self.ix[0]   ][ self.ix[1] ] = matrix[ self.ix[0]+1 ][ self.ix[1] ]
            matrix[  self.ix[0]+1 ][ self.ix[1] ] = self.EMPTY_SPACE
            newIx = (self.ix[0]+1 ,  self.ix[1] )
            return Puzzle( matrix, newIx )
        if movement is Movements.LEFT:
            matrix[  self.ix[0]   ][ self.ix[1] ]   = matrix[ self.ix[0] ][ self.ix[1]-1 ]
            matrix[  self.ix[0]   ][ self.ix[1]-1 ] = self.EMPTY_SPACE
            newIx = (self.ix[0]   ,  self.ix[1]-1 )
            return Puzzle( matrix, newIx )
        if movement is Movements.RIGHT:
            matrix[  self.ix[0]   ][ self.ix[1] ]   = matrix[ self.ix[0] ][ self.ix[1]+1 ]
            matrix[  self.ix[0]   ][ self.ix[1]+1 ] = self.EMPTY_SPACE
            newIx = (self.ix[0]   ,  self.ix[1]+1 )
            return Puzzle( matrix, newIx )
        return Puzzle( matrix, (self.ix[0],self.ix[1]) ) #We do not know that movement, return a copy

    def canMove(self, movement): #self.ix => (row,col)
        if movement is Movements.UP:
            return self.ix[0]-1 > 0             #Existe esa row
        if movement is Movements.DOWN:
            return self.ix[0]+1 < self.shape[0] #Existe esa row
        if movement is Movements.LEFT:
            return self.ix[1]-1 > 0             #Existe esa col
        if movement is Movements.RIGHT:
            return self.ix[1]+1 < self.shape[1] #Existe esa col
        return False #No conocemos ese movimiento

'''
edoInicial: Un estado inicial del 8‐puzzle. El estado inicial es una lista de listas, donde cada lista interna contendrá 3 dígitos del 0 al 8 representa un renglón del 8‐puzzle. El espacio será representado por el número 0. Por ejemplo, el estado mostrado en la figura 1 se representará con la lista: [[1, 2, 3], [4, 5, 6], [7, 8, 0]].
edoFinal  : Un estado meta del 8‐puzzle. Representado igual que el estado inicial (es decir, con una lista de listas de dígitos)
algoritmo : El tipo de algoritmo a utilizar. Esta es una variable entera, si su valor es 0, se debe usar BFS y si es 1, DFS.
'''
def busquedaNoInformada(edoInicial, edoFinal, DFS=True): #0 BFS o 1 para DFS
    root       = PuzzleNode( Puzzle(edoInicial) )  #Inicializar raiz
    finalState = Puzzle(edoFinal)
    answer     = None

    structure = deque()    #Deque is faster for popings but slower for random accesses
    structure.append(root)
    getNode = lambda struc: struc.pop() if DFS else struc.popleft() 
    visited = set()

    while structure: #Not Empty
        currentNode = getNode(structure)    #This function is defined using the DFS flag
        if currentNode.state == finalState: #Node is the wrapper, we only compare the puzzle
            answer = currentNode            #Save the answer
            break                           #We have finished
        if currentNode.state in visited: continue  
        structure.extend( currentNode.getCombinations() ) #Add the elements to the structure
        visited.add( currentNode.state )

    return [] if answer is None else answer.backTrack() #Answer is a node pointing to more nodes

if __name__ == '__main__':
    edoInicial  = [[0, 1, 2], [4, 5, 3], [7, 8, 6]]
    edoFinal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]] 

    steps = busquedaNoInformada(edoInicial, edoFinal, 0) # puede llamarse con 1
    print (steps)
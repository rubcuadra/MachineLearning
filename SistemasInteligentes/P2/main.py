#RUBEN CUADRA A01019102
from queue import PriorityQueue as PQ
from enum import Enum

class Movements(Enum):
    UP    = "U"
    DOWN  = "D"
    LEFT  = "L"
    RIGHT = "R"

#For backtrack and sum of costs
class PuzzleEdge(object):
    def __init__(self, tag, val=1):
        super(PuzzleEdge, self).__init__()
        self.tag      = tag
        self.val      = val

class PuzzleNode(object): #Wrapper for Tree functionalities
    @classmethod
    def setHeuristic(cls, f): cls.heuristic = f

    @staticmethod
    def heuristic(state):     return 0

    def __init__(self, _puzzle, parentNode=None, edge=None):
        super(PuzzleNode, self).__init__()
        self.state      = _puzzle
        self.parentNode = parentNode
        self.edge       = edge if parentNode!=None else None #It is the action/value that connects with the parentNode
        self.updateVal() #According to the heuristic

    #f(n) = g(n) + h(n) 
    #g(n) is the cost from root to this node ; Maybe pass from node instead of calculate it? it'd be more efficient (code won't look cool tho)
    #h(n) is obtained with the heuristic
    def updateVal(self):  
        self.val = self.getCostToRoot() + self.__class__.heuristic( self.state )

    def getCombinations(self): #Iterator
        for m in Movements:    #TRY all existing movements
            if self.state.canMove( m ):
                yield PuzzleNode( self.state.getMovement(m), parentNode=self, edge=PuzzleEdge(m) ) #Move returns a Puzzle object
        
    def getCostToRoot(self):#Returns the sum of costs
        if self.parentNode is None: return 0
        return self.parentNode.getCostToRoot() + self.edge.val

    def backTrack(self):    #Return the tags as array
        if self.parentNode is None: return []
        return self.parentNode.backTrack() + [self.edge.tag.value] #self.edge.tag is an enum, it has a .value

    def __lt__(self,other): #For priority queue, would be better to use edge and val, it uses heuristic
        return self.val < other.val #self.edge.val + self.val #Costo real + Heuristica

class Puzzle(object):
    EMPTY_SPACE = 0

    def __iter__(self):
        self.i = 0
        return self

    def __next__(self):
        if self.i >= self.shape[0]: 
            raise StopIteration
        self.i += 1
        return self.board[self.i-1]
    
    def __init__(self, val, ix = None):
        super(Puzzle, self).__init__()
        #Convert from [[0, 1, 2], [4, 5, 3], [7, 8, 6]] to ((0, 1, 2), (4, 5, 3), (7, 8, 6))
        self.shape = ( len(val), len(val[0]) )
        self.board = val  #No need of tuples since we are not hashing
        self.ix    = None #Empty space index

        #Find the empty space
        if ix is None: #If they don't give an ix we need to find it by searching in matrix
            for i,row in enumerate(val):
                for j,cell in enumerate(row):
                    if cell == self.EMPTY_SPACE:
                        self.ix = (i,j)     #Save and break
                        break
                if self.ix != None:         #Alerady got it
                    break
        else:
            self.ix = ix #is a tuple of (row,column)
        
    def __eq__(self, other):
        return self.board == other.board #It compares matrix boards

    def __hash__(self):
        return hash(self.board) #ERROR because we are using lists, it was working for tuples

    @staticmethod
    def getManhattanDistanceHeuristic(finalState):
        finalPositions = {}
        for i,row in enumerate(finalState):
            for j,cell in enumerate(row):
                finalPositions[cell] = (i,j) 
        def h(s):
            distance = 0
            for i,row in enumerate(s):
                for j,cell in enumerate(row):
                    finalPos = finalPositions[cell]
                    distance += abs(finalPos[0] - i) + abs(finalPos[1] - j)
            return distance
        return h

    @staticmethod
    def getWrongCellsHeuristic(finalState):
        finalPositions = {}
        for i,row in enumerate(finalState):
            for j,cell in enumerate(row):
                finalPositions[cell] = (i,j) 

        def h(s):
            wrong = 0
            for i,row in enumerate(s):
                for j,cell in enumerate(row):
                    if i==finalPositions[cell][0] and j==finalPositions[cell][1]: continue
                    wrong+=1
            return wrong
        return h

    def getMovement(self,movement): #We assume that canMove was called already 
        matrix = list( map(list, self.board) ) #Copy current state
        
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
heuristic : 1 or 0, es la heuristica a usar
'''
def busquedaAstar(edoInicial, edoFinal, heuristic=1): #0 Cuadros o 1 para Manhattan Distance
    if   heuristic:   PuzzleNode.setHeuristic( Puzzle.getManhattanDistanceHeuristic( edoFinal ) )
    else:             PuzzleNode.setHeuristic( Puzzle.getWrongCellsHeuristic( edoFinal ) )

    root       = PuzzleNode( Puzzle(edoInicial) )  #Inicializar raiz
    finalState = Puzzle(edoFinal)
    answer     = None

    structure = PQ()    
    structure.put(root)
    
    while not structure.empty():     
        currentNode = structure.get()   
        if currentNode.state == finalState: #Node is the wrapper, we only compare the puzzle
            answer = currentNode            #Save the answer
            break                           #We have finished
        for combination in currentNode.getCombinations(): structure.put( combination  ) #Add the elements to the structure
    return [] if answer is None else answer.backTrack() #Answer is a node pointing to more nodes

if __name__ == '__main__':
    edoInicial  = [[0, 1, 2], [4, 5, 3], [7, 8, 6]]
    edoFinal = [[1, 2, 3], [4, 5, 6], [7, 8, 0]] 
    steps = busquedaAstar(edoInicial, edoFinal, 0) # puede llamarse con 1
    print (steps)

from reversi import ReversiBoard,CELLS_IX

class BoardNode(object):
    def __init__(self, board, player, cell=None, score=None,):
        super(BoardNode, self).__init__()
        self.board         = board
        self.player        = player   
        self.children      = [] 
        self.score         = self.getScore() if not score else score
        if cell: self.cell = cell #Backtracking Ej. A1 , B3, D2

    #Get all possible children nodes given one player
    def childrenIterator(self, player=None): 
        if player is None: player = self.player
        for cell in CELLS_IX:
            r = self.board.throw(player,cell)
            if r: #Create new board
                newBoard = ReversiBoard( self.board.value )
                newBoard.doUpdate(player,r)
                yield BoardNode(newBoard,player,cell)

    #Call iterator with the same player or with the opponent
    def updateChildren(self, change_player = False):
        player = ReversiBoard.getOpponent(self.player) if change_player else self.player
        self.children = [ bn for bn in self.childrenIterator(player)]

    #Heuristic
    def getScore(self): #Quiza se deberia mejorar
        return self.board.score(self.player)

    #For Max and Min
    def __lt__(self,other):
        return self.score < other.score

    @staticmethod
    def MM(node,level=0): #Regresa scores
        if level <= 0:    return node.score
        node.updateChildren(change_player=True) #Expandir y sacar Max o Min de los hijos
        if node.children: 
            if level%2 == 0: #MAX
                return max( [ BoardNode.MM(c,level-1) for c in node.children ] ) 
            else: #MIN   
                return min( [ BoardNode.MM(c,level-1) for c in node.children ] ) 
        return node.score #Se intento expandir pero ya no se puede

class Agent(object):
    @staticmethod
    def getBestMovement(board, player, level=1): #Level 1 = Expande 1 tiro nuestro y 1 de oponente
        root = BoardNode(board,player)    #Root
        root.updateChildren()             #First Layer, get Max
    
        #Get Max but save the movement
        mx = ("D6", float("-inf"))
        for ch in root.children:
            score = BoardNode.MM(ch, 1 + (level-1)*2 )
            print(score)
            if score > mx[1]: mx = ( ch.cell, score )    
        return mx[0]

from enum import Enum
from movements import OnitamaCards

class OnitamaBoard():
    """
        board => matrix of 5x5 with MASTERS and STUDENTS
        cards => Array of 3 elements
            1) set of 2 elements with BLUE_PLAYER cards
            1) set of 2 elements with RED_PLAYER cards
            1) set of 1 elements with STAND_BY card
    """
    BLUE,RED = 0,1
    #Value in matrix, Symbol to Print
    BLUE_MASTER  = "B"
    RED_MASTER   = "R"
    BLUE_STUDENT = "b"
    RED_STUDENT  = "r"
    EMPTY_CHAR   = " "

    def __init__(self, board=[], cards=[], extras={}):
        if board: 
            self.board = board
            self.cards = cards
            self._blue = extras["b"] 
            self._red  = extras["r"] 
            self._blue_is_alive = extras["mb"] 
            self._red_is_alive = extras["mr"] 
            self._blue_master_pos = extras["mbp"] 
            self._red_master_pos  = extras["mrp"] 
            self._blue_pos = extras["bp"] 
            self._red_pos  = extras["rp"] 
        else: #Initial board
            self.board = [
                [self.RED_STUDENT] *5,
                [self.EMPTY_CHAR]  *5,
                [self.EMPTY_CHAR]  *5,
                [self.EMPTY_CHAR]  *5,
                [self.BLUE_STUDENT]*5,
            ]
            self.board[0][2] = self.RED_MASTER
            self.board[4][2] = self.BLUE_MASTER

            cards = OnitamaCards.random_keys(5)
            self.cards = [
                set(cards[0:2]), #BLUE
                set(cards[2:4]), #RED
                cards[-1],
            ]
            #For scores and game over, better to have it like this than calculating it on each move
            self._blue = 5
            self._red  = 5
            self._blue_is_alive = True
            self._red_is_alive = True
            self._blue_master_pos = (4,2)
            self._red_master_pos  = (0,2)
            self._blue_pos = set(( (4,0),(4,1),(4,3),(4,4) ))
            self._red_pos  = set(( (0,0),(0,1),(0,3),(0,4) ))

    #Se deberia dar por el tablero + cartas
    # def __hash__(self):
    #     return hash( tuple( map(tuple, self.board) ) )

    def __getitem__(self,i): return self.board[i]

    @classmethod
    def getOpponent(cls,player):
        return cls.RED if player is cls.BLUE else cls.BLUE

    @classmethod
    def isPlayer(cls,player,token):
        if player is cls.BLUE: return token == cls.BLUE_MASTER or token == cls.BLUE_STUDENT
        return token == cls.RED_MASTER or token == cls.RED_STUDENT

    def canMove(self, player, fromCell, card, toCell):
        fromRow, fromCol = fromCell
        toRow,     toCol = toCell
        #Checar indices
        if fromRow > -1 and toRow > -1 and fromCol > -1 and toCol > -1 and fromRow < 5 and toRow < 5 and fromCol < 5 and toCol < 5:
            #Checar indices origen y destino (que no se salgan)
            #Origin is player's token
            #Destin is NOT player's token
            #Player has that card
            if self.isPlayer(player,self[fromRow][fromCol]) and not self.isPlayer(player,self[toRow][toCol]) and card in self.cards[player]:
                mov = (toCell[0]-fromCell[0],toCell[1]-fromCell[1]) if player is self.BLUE else (fromCell[0]-toCell[0],fromCell[1]-toCell[1])
                return mov in OnitamaCards[card]
        return False
    
    #Should call canMove before, otherwise we'll have buggs
    #Returns a new Board
    def move(self, player, fromCell, card, toCell):
        board = list( map(list, self.board) )
        #Move token
        dest = board[toCell[0]][toCell[1]]
        orig = board[fromCell[0]][fromCell[1]]
        board[toCell[0]][toCell[1]] = board[fromCell[0]][fromCell[1]]
        board[fromCell[0]][fromCell[1]] = self.EMPTY_CHAR
        #Update Cards
        B,R,SB = self.getCards()
        B,R    = set(B),set(R)     #Work on copies
        
        r,b   = self._red, self._blue                         #int
        mr,mb = self._red_is_alive, self._blue_is_alive       #bool
        mbp,mrp = self._blue_master_pos, self._red_master_pos #tuple
        bp, rp = set(self._blue_pos), set(self._red_pos)      #set of tuples
        if player is self.BLUE: 
            #Add card
            B.add(SB)
            B.remove(card)
            #Update master position
            if orig == self.BLUE_MASTER: mbp = toCell 
            else: #Update students positons
                bp.add(toCell)
                bp.remove(fromCell)
            #Update pieces count
            if   dest == self.RED_MASTER: r,mr = r-1,False
            elif dest == self.RED_STUDENT:r -= 1
        else:
            R.add(SB)
            R.remove(card)
            #Update master position
            if orig == self.RED_MASTER: mrp = toCell 
            else: #Update students positons
                rp.add(toCell)
                rp.remove(fromCell)
            #Update pieces count
            if   dest == self.BLUE_MASTER:b,mb = b-1,False
            elif dest == self.BLUE_MASTER:b -= 1
        SB = card                  #Send movement to Stand By
        cards = [ B,R,SB ]         #New cards
        #Return new Board
        return OnitamaBoard(board,cards,{"b":b,"r":r,"mb":mb,"mr":mr,"mrp":mrp,"mbp":mbp,"bp":bp,"rp":rp})

    def isGameOver(self):
        f = self[0][2] == self.BLUE_MASTER or self[4][2] == self.RED_MASTER
        if not f: #If none arrived to the other side check if a master is gone
            return not (self._red_is_alive and self._blue_is_alive)
        return True 

    #BLUE,RED,STAND_BY = getCards()
    def getCards(self): return self.cards

    def __str__(self):
        print("BLUE")
        for c in self.cards[self.BLUE]: print("\t",c)
        print("RED")
        for c in self.cards[self.RED]: print("\t",c)
        print("Extra\n\t", self.cards[2],"\n")
        toRet = ""
        for i,row in enumerate( self.board ) :
            toRet += "|"
            for cell in self.board[i]: toRet += f"{cell}|"
            toRet += "\t\t|"
            for cell in self.board[-(i+1)]: toRet += f"{cell}|"
            toRet += "\n"
        return toRet

if __name__ == '__main__':
    from random import seed
    seed(0)
    board = OnitamaBoard()
    if board.canMove( board.RED, (0,2), "RABBIT", (1,1) ) :
        newBoard = board.move( board.RED, (0,2), "RABBIT", (1,1) )
        print(newBoard) 

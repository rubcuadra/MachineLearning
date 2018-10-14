from enum import Enum
from movements import OnitamaCards
from random import seed

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

    def __init__(self, board=None, cards=None):
        if board: 
            self.board = board
            self.cards = cards
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

    def __getitem__(self,i): return self.board[i]

    @classmethod
    def isPlayer(cls,player,token):
        if player is cls.BLUE: return token == cls.BLUE_MASTER or token == cls.BLUE_STUDENT
        return token == cls.RED_MASTER or token == cls.RED_STUDENT

    def canMove(self, player, fromCell, card, toCell):
        fromRow, fromCol = fromCell
        toRow,     toCol = toCell
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
        board[toCell[0]][toCell[1]] = board[fromCell[0]][fromCell[1]]
        board[fromCell[0]][fromCell[1]] = self.EMPTY_CHAR
        #Update Cards
        B,R,SB = self.getCards()
        B,R    = set(B),set(R)     #Work on copies
        if player is self.BLUE: 
            B.add(SB)
            B.remove(card)
        else:
            R.add(SB)
            R.remove(card)
        SB = card                  #Send movement to Stand By
        cards = [ B,R,SB ]         #New cards
        #Return new Board
        return OnitamaBoard(board,cards)

    
    def isGameOver(self):
        f = self[0][2] == self.BLUE_MASTER or self[4][2] == self.RED_MASTER
        if not f: #If none arrived to the other side check if a master is gone
            red, blue = False, False
            for row in self:
                for cell in row:
                    red  |= cell==self.RED_MASTER
                    blue |= cell==self.BLUE_MASTER
                    if red and blue: break #Both are alive
            return not (red and blue) #If 1 is gone then it is over
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
    board = OnitamaBoard()
    if board.canMove( board.RED, (0,3), "RABBIT", (1,2) ) :
        newBoard = board.move( board.RED, (0,3), "RABBIT", (1,2) ) 
        print(newBoard)

from enum import Enum
from randomDict import RandomDict as RD

#Row, Column
DR   = ( 1, 1)
DL   = ( 1,-1)
UR   = (-1, 1)
UL   = (-1,-1)
U    = (-1, 0)
UU   = (-2, 0)
D    = ( 1, 0)
R    = ( 0, 1)
RR   = ( 0, 2)
L    = ( 0,-1)
LL   = ( 0,-2)
LLU  = (-1,-2)
RRU  = (-1, 2)

#Para BLUE (Player que va de abajo hacia arriba)
OnitamaCards = RD({
    #Adjacent
    "MANTIS"   : ( UL, UR, D ),
    "OX"       : ( U , R,  D ),
    "HORSE"    : ( U , L,  D ),
    "EEL"      : ( UL, DL, R ),
    "CRANE"    : ( DL, DR, U ),
    "BOAR"     : ( L , R,  U ),
    "COBRA"    : ( L , UR, DR),
    "MONKEY"   : ( UL, UR, DL, DR),
    "ELEPHANT" : ( UL, UR, L , R),
    "ROOSTER"  : ( DL, UR, L , R),
    "GOOSE"    : ( UL, DR, L , R),
    #Jumps
    "TIGER"    : ( UU, D     ),
    "RABBIT"   : ( DL, UR, RR),
    "FROG"     : ( DR, UL, LL),
    "CRAB"     : ( LL, U , RR) ,
    "DRAGON"   : ( DL, DR, LLU, RRU),
})

class OnitamaBoard():
    """
        board => matrix of 5x5 with MASTERS and STUDENTS
        cards => Array of 3 elements
            1) Tuple of size 2 with BLUE_PLAYER cards
            2) Tuple of size 2 with  RED_PLAYER cards
            3) Tuple of size 1 with STAND_BY    card
    """
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
                cards[0:2],
                cards[2:4],
                cards[-1]
            ]

    def move(self, fromCell, card, toCell):
        fromRow, fromCol = fromCell
        toRow,     toCol = toCell
        player = self.getPlayer( fromRow, fromCol )

    def __str__(self):
        print("BLUE\t",self.cards[0][0],' - ',self.cards[0][1])
        print("RED \t",self.cards[1][0],' - ',self.cards[1][1])
        print("Extra \t",self.cards[2],"\n")
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
    print(board)


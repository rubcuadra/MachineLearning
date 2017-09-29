#Ruben Cuadra
#Seung Hoon Lee
import numpy as np
from copy import copy
from random import random,randint
from sys import maxsize

class BoardNode():
    #Multiplicadores de scores
    ourPlayer   = 1
    otherPlayer = -1
    nobody      = 0

    def __getitem__(self,i):
         return self.children[i]

    def __init__(self, data, player=nobody, thrown=None):
        self.board = data
        self.player = player
        self.col = thrown
        self.children = []
    
    def __str__(self):
        s = ""
        for c in self.children:
            s += str(c.board) + "\n"
        return str(self.board) + "->\n" + str(s)

    def add_children_boards(self, boardsArray, player):
        for i,board in enumerate(boardsArray):
            self.children.append( BoardNode(board,player,i) )

    #Devuelve el puntaje maximo de este Nodo sumandole/restandole sus hijos
    def getMaxScore(self):        #Nos da el mas grande 
        if not self.board:          #Tablero None
            return 0          #Neutro aditivo
        if not self.children: #Ya se acabo, solo regresar su valor
            return self.board.score*self.player
        else:
            return max( c.getMaxScore()+self.board.score*self.player for c in self.children )

    #Nos devuelve el indice del hijo con mejor score
    #y su Score
    def getMax(self): 
        maxi = 0
        maxScore = Board.minScore
        for i,c in enumerate(self.children):     #Nodos Hijos
            if c.board is None: continue
            currentMax = c.getMaxScore()             #Obtener el mayor puntaje posible de este node
            if currentMax > maxScore:
                maxScore = currentMax
                maxi = i
        return (maxi,maxScore)

    #Empieza a agregar hijos al nodo, us es para saber si usa
    #el ourDisk o el enemyDisk, esto afecta los prints y 
    #para calcular puntajes, nos da + a nosotros y - cuando es del enemigo
    def addDepth(self, depth, ourDisk, enemyDisk, us):
        if not depth or not self.board: return
        if us:
            player = BoardNode.ourPlayer 
            disk   = ourDisk
        else:
            player = BoardNode.otherPlayer
            disk   = enemyDisk
        self.add_children_boards(self.board.getCombinations(disk),player)
        for child in self.children: 
            child.addDepth(depth-1,ourDisk,enemyDisk,not us)

class Board():
    minScore = -1*maxsize
    def __init__(self,nativeBoard=None):
        self.val = None
        self.score = self.minScore
        if nativeBoard:
            self.val = np.flip(np.array(nativeBoard),axis=0)#Fix para que salga como deberia

    def __getitem__(self,i):
         return self.val[i]

    def __str__(self):
        return str(self.val)

    def __gt__(self, other):
        return self.score > other.score

    def __ge__(self, other):
        return self.score >= other.score

    def __copy__(self):
        newone = type(self)()
        newone.val = self.val.copy()
        return newone

    def getCombinations(self, diskToThrow): #Regresamos copias
        gc = []
        for i in range(self.val.shape[1]):            
            gc.append(Board.throw(copy(self),i,diskToThrow))
        return gc 
    
    def throw(brd,col_i,disk):
        ix = np.where(brd.val[:,col_i]==0)[0]
        if len(ix)==0:return None                    #Ya no se puede tirar ahi
        brd.val[:,col_i][ix[-1]] = disk              #Hacer el tiro
        brd.score = Board.getBoardScore(brd,disk,col_i) #Actualizar el score
        return brd                                   #Regresar tablero modificado

    def getSize(self):
        return self.val.shape

    #Regresa la celda o None
    def g(self,row,column):
        if row<0 or column<0:return None
        try:    return self[row][column]
        except: return None

    def getBoardScore(_board, ourDisk, movedColumn):
        if not _board: return Board.minScore  #Regresar el minimo   
        
        rows,columns = _board.getSize() #Saber si ya nos pasamos de rows o columns
        #movedColumn = movedColumn      #Es el que nos pasaron
        movedRow     = _board.val[:,movedColumn].nonzero()[0][0]

        accumPoints = 100
        #Celdas para validar, pueden ser None
        current = _board[movedRow][movedColumn]
        der = _board.g(movedRow,movedColumn+1)
        izq = _board.g(movedRow,movedColumn-1)
        up = _board.g(movedRow-1,movedColumn)
        dwn = _board.g(movedRow+1,movedColumn)
        #Diagonales directas - grado 1, solo se cambia el grado
        upDer  = _board.g(movedRow-1,movedColumn+1)
        dwnDer = _board.g(movedRow+1,movedColumn+1)
        upIzq  = _board.g(movedRow-1,movedColumn-1)
        dwnIzq = _board.g(movedRow+1,movedColumn-1)

        #Checar si la primera fila no hay nada en las 3 columnas del centro
        if any ( [movedColumn == 2 and movedRow == 5, movedColumn == 3 and movedRow == 5, movedColumn == 4 and movedRow == 5] ):
            accumPoints += 500

        #checar si dos puntos se llegan a pegar
        if any ( [der == current, izq == current, up == current, upDer == current, dwnDer == current, upIzq == current, dwnIzq == current] ):
            accumPoints += 1000

        #checar si tres puntos se llegan a pegar
        if any ( [der == current and izq == current, up == current and dwn == current, upDer == current and dwnIzq == current, upIzq == current and dwnDer == current]):
            accumPoints += 1000
        
        #checar si se llega a formar una T
        if any ( [der == current and izq == current and dwn == current,
                    dwnIzq == current and dwnDer == current and dwn == current,
                    upIzq == current and dwnIzq == current and izq == current,
                    upDer == current and dwnDer == current and der == current,
                    dwnIzq == current and upIzq == current and dwnDer == current,
                    dwnIzq == current and upDer == current and dwnDer == current,
                    upIzq == current and dwnDer == current and upDer == current,
                    dwnIzq == current and upIzq == current and upDer == current
                    ] ):
            accumPoints += 8000

        #Enemy disk
        enemyDisk = 1 if ourDisk is 2 else 2 #Bien naco, np buscar diferentes a 0 y ourdisk
        
        #checar si se llega a formar una T
        if any ( [der == enemyDisk and izq == enemyDisk and dwn == enemyDisk,
                    dwnIzq == enemyDisk and dwnDer == enemyDisk and dwn == enemyDisk,
                    upIzq == enemyDisk and dwnIzq == enemyDisk and izq == enemyDisk,
                    upDer == enemyDisk and dwnDer == enemyDisk and der == enemyDisk,
                    dwnIzq == enemyDisk and upIzq == enemyDisk and dwnDer == enemyDisk,
                    dwnIzq == enemyDisk and upDer == enemyDisk and dwnDer == enemyDisk,
                    upIzq == enemyDisk and dwnDer == enemyDisk and upDer == enemyDisk,
                    dwnIzq == enemyDisk and upIzq == enemyDisk and upDer == enemyDisk,
                    der == enemyDisk and der_der == enemyDisk and dwnDer == enemyDisk,
                    izq == enemyDisk and izq_izq == enemyDisk and dwnIzq == enemyDisk,
                    der == enemyDisk and upDer == enemyDisk and der_der == enemyDisk,
                    izq == enemyDisk and izq_izq == enemyDisk and upIzq == enemyDisk,
                    dwn == enemyDisk and dwn_dwn == enemyDisk and dwnDer == enemyDisk,
                    dwn == enemyDisk and dwn_dwn == enemyDisk and dwnIzq == enemyDisk,
                    dwnDer == enemyDisk and dwnDer_dwnDer == enemyDisk and dwn_dwn == enemyDisk,
                    upIzq == enemyDisk and upIzq_upIzq == enemyDisk and izq_izq == enemyDisk,
                    dwnIzq == enemyDisk and dwnIzq_dwnIzq == enemyDisk and dwn_dwn == enemyDisk,
                    upDer == enemyDisk and upDer_upDer == enemyDisk and der_der == enemyDisk,
                    dwnDer == enemyDisk and dwnDer_dwnDer == enemyDisk and der_der == enemyDisk,
                    izq_izq == enemyDisk and dwnIzq == enemyDisk and dwn_dwn == enemyDisk,
                    dwnIzq == enemyDisk and dwnIzq_dwnIzq == enemyDisk and izq_izq == enemyDisk,
                    der_der == enemyDisk and dwnDer == enemyDisk and dwn_dwn == enemyDisk] ):
            accumPoints += 4000
        
        #Numero de movimientos
        accumPoints -= np.count_nonzero(_board.val)/2
        return accumPoints #Regresar Puntaje acumulado

def getBestColToPlay(nativeBoard,disk,totalColumns=7,depth=5):
    _board = Board(nativeBoard) #Crear un tablero nuestro
    root = BoardNode( _board )
    #Nuestro disco y el enemigo, genera N profundidad
    root.addDepth(depth,disk,disk+1,us=True) 
    #Obtener indice del hijo con mayor peso
    colToThrow,colScore = root.getMax()
    # print(root)                #Imprime su board y sus hijos
    # print(colToThrow,colScore) #Imprime col designada y el puntaje ahi
    return colToThrow



#Usar A* con heuristicas de distancia manhattan y cuadrosFueraDelLugar
from queue import PriorityQueue as PQ

#For the Priority queue, if we pass just a tuple it affects the outcome 
class PrioritizedItem: 
    @classmethod
    def setHeuristic(cls, f): 
        cls.heuristic = f

    @staticmethod
    def heuristic(state): 
        return 0

    def testH(self):
        print( self.__class__.heuristic(self.val) )

    def __init__(self,val):
        self.val = val

    def __lt__(self,other):
        return self.val > other.val

    def __str__(self):
        return str(self.val)

if __name__ == '__main__':
    q = PQ()
    q.put( PrioritizedItem(10) )
    q.put( PrioritizedItem(4) )
    q.put( PrioritizedItem(17) )

    while not q.empty():
        n = q.get()
        print(n)
    
    PrioritizedItem.setHeuristic(lambda x:3) 
    a = PrioritizedItem(1)
    
    a.testH()
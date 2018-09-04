from queue import PriorityQueue as PQ

#For the Priority queue, if we pass just a tuple it affects the outcome 
class PrioritizedItem: 
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
    q.put( PrioritizedItem(1) )

    while not q.empty():
        n = q.get()
        print(n)
    
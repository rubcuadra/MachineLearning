import matplotlib.pyplot as plt
import numpy as np
from queue import PriorityQueue as PQ
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from matplotlib.ticker import MaxNLocator
import csv

#Recibe como parametro el path a un archivo csv
#con +2 columnas, "x,x1,x2,...xN,y" dond ela ultima es el valor en Y
def getDataFromFile(filename):  
    val = []
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i in reader:
            val.append( [float(j) for j in i] )
    return np.array(val)

def drawPoints(points,tags,Xtest,Ytest):
    for p,tag in zip(points,tags):
        if tag: plt.scatter(p[0],p[1],marker="x", label="True",color="black")
        else:   plt.scatter(p[0],p[1],marker="o", label="False",color="black")
    
    #We change color to red for the TEST data
    for p,tag in zip(Xtest,Ytest):
        if tag: plt.scatter(p[0],p[1],marker="x", label="True", color="red")
        else:   plt.scatter(p[0],p[1],marker="o", label="False", color="red" )
    
    plt.show()   # Show the plot
#METRICS
def getEuclideanDistance(p1,p2): return np.linalg.norm(p1-p2)
def manhattanMetric(p1,p2): return sum( abs(d1-d2) for d1,d2 in zip(p1,p2) )
def maxNormMetric(p1,p2): return   max( abs(d1-d2) for d1,d2 in zip(p1,p2) )
def LKMetric(p1,p2,k=3): return    sum( abs(d1-d2)**k for d1,d2 in zip(p1,p2) )**(1/k)

#For the Priority queue, if we pass just a tuple it affects the outcome 
class DL: 
    def __init__(self,distance,label):
        self.distance = distance
        self.label = label
    def __lt__(self, other):
        return self.distance < other.distance
    def __str__(self):
        return f'{self.distance} {self.label}'
        
#Distance function
#points Nx2 
#labels 2x1 , floats with 1.0 or 0.0
def KNN(k,center,points,labels,dF,kernel):
    q = PQ()    #Create PriorityQueue
    for point,label in zip(points,labels): 
        distance = dF(point,center)   #Calculate distance with the metric
        q.put( DL(distance,label) )   #Add the distance with the tag to the queue

    dLabels = { c:0 for c in np.unique(labels) } #get unique labels
    
    while (not q.empty()) and k>0: #Iterate on the priority queue k times
        dLabels[q.get().label] += kernel(distance)   # Add 1 is without kernel
        k-=1

    return max(dLabels, key=dLabels.get) #Max from the count
    
def plotAccuracy(accVector):
    plt.plot( [i for i in range(1,len(accVector)+1)],accVector,label="KNN Accuracy")
    plt.xlabel('K')
    plt.ylabel('Probability')
    plt.axis([1,len(accVector)+1, 0, 1.0])
    plt.grid(True)
    plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True)) #Integer Xs
    plt.legend() # Add a legend
    plt.show()   # Show the plot

#KERNELS
#https://en.wikipedia.org/wiki/Kernel_(statistics)
def epanechnikovKernel(u): return (3/4)*(1-u**2)
def counter(u): return 1

def Test(func,k,metric,trainingData,trainingTags,points,tags,kernel):
    corrects = 0
    preds = []
    for point,tag in zip(points,tags):
        result = func(k,point,trainingData,trainingTags,metric,kernel) #Returns the calculated tag
        preds.append(result)
        if result == tag: corrects+=1
    
    acc = corrects/len(points)
    # print(f"\t\t{k}NN : {corrects}/{len(points)} = {acc}")
    return acc,preds

def crossVal(data,tags,intervals=5):    
    inters = int( len(data)/intervals )

    for i in range(intervals):
        _f = inters*i
        _t = inters*(i+1)
        
        if i == 0:
            trainX = data[_t:]
            trainY = tags[_t:]
        elif i == intervals-1:
            trainX = data[:_f]
            trainY = tags[:_f]
        else:
            trainX = np.concatenate((data[:_f] , data[_t:]), axis=0)
            trainY = np.concatenate((tags[:_f] , tags[_t:]), axis=0)
            
        yield (trainX,data[_f:_t],trainY,tags[_f:_t])


if __name__ == '__main__':
    vals = getDataFromFile("chips.txt")
    X,tags = vals[:,0:-1] , vals[:,-1]
    
    metrics = [getEuclideanDistance,manhattanMetric,maxNormMetric,LKMetric] #LkNorm
    kernels = [counter, epanechnikovKernel] 
    
    for X_train,X_test,Y_train,Y_test in crossVal(X,tags,intervals=5):
        # X_train,X_test,Y_train,Y_test = train_test_split(X,tags,test_size=0.1)
        
        bestK,accuracy,ker,metr,_F = 1,0.0,kernels[0],metrics[0],0
        for m in metrics:
            # print(m.__name__)
            for kernel in kernels:
                # print('\t',kernel.__name__)
                for k in range(20):
                    k +=1 #Starting in 1 and finishing in 20
                    F,predictions = Test(KNN,k,m,X_train,Y_train,X_test,Y_test,kernel) 
                    if F>accuracy:
                        bestK = k
                        accuracy = F
                        ker = kernel
                        metr = m
                        _F = f1_score(Y_test, predictions, average='macro')

        print (f'Best results: {bestK}NN {accuracy} F:{_F} {ker.__name__} {metr.__name__}')
        
    
    # drawPoints(X_train,Y_train,X_test,Y_test)
    
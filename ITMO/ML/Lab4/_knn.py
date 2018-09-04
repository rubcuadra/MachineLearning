from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap
from matplotlib.ticker import MaxNLocator
from sklearn.metrics import accuracy_score
from queue import PriorityQueue as PQ
import matplotlib.pyplot as plt
import numpy as np
import csv

cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])

#Recibe como parametro el path a un archivo csv
#con +2 columnas, "x,x1,x2,...xN,y" dond ela ultima es el valor en Y
def getDataFromFile(filename,shuffle=False):  
    val = []
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i in reader:
            val.append( [float(j) for j in i] )
    val = np.array(val)
    if shuffle: np.random.shuffle(val)
    return val[:,0:-1] , val[:,-1] #Y is the last column, else X

#METRICS
def getEuclideanDistance(p1,p2): return np.linalg.norm(p1-p2)
def manhattanMetric(p1,p2): return sum( abs(d1-d2) for d1,d2 in zip(p1,p2) )
def maxNormMetric(p1,p2): return   max( abs(d1-d2) for d1,d2 in zip(p1,p2) )
def LKMetric(p1,p2,k=3): return   sum( abs(d1-d2)**k for d1,d2 in zip(p1,p2) )**(1/k)

#For the Priority queue, if we pass just a tuple it affects the outcome 
class DL: 
    def __init__(self,distance,label):
        self.distance = distance
        self.label = label
    def __lt__(self, other):
        return self.distance < other.distance
    def __str__(self):
        return f'{self.distance} {self.label}'

#KERNELS
#https://en.wikipedia.org/wiki/Kernel_(statistics)
def epanechnikovKernel(u): return (3/4)*(1-u**2)
def counter(u): return 1
def exp(u,s=10): return np.exp(-(u-s))

class KNN():
    def __init__(self,k=1,metric=getEuclideanDistance,kernel=counter):
        self.k = k
        self.metric = metric
        self.kernel = kernel

    def fit(self,data,labels):
        self.data       = data
        self.labels     = labels
        self.labelCount = { c:0 for c in np.unique(labels) } #get unique labels        

    def predict(self, X):
        res = []
        for v in X:
            q = PQ()    #Create PriorityQueue for that Point
            for point,label in zip(self.data, self.labels):     
                distance = self.metric(point,v)   #Calculate distance with the metric
                q.put( DL(distance,label) )       #Add the distance with the tag to the queue    
            counter = self.labelCount.copy()
            k = self.k
            while (not q.empty()) and k>0: #Iterate on the priority queue k times
                counter[q.get().label] += self.kernel(distance) ; k-=1
            res.append(max(counter, key=counter.get)) #Max from the count
        return np.array(res)

    #PLOTS  
    #   black - trainning data
    #   red   - missclasified data
    #   green - correct classification
    def plot(self, data,tags):
        #PLOT AREAS
        x_min, x_max = data[:, 0].min() - 1, data[:, 0].max() + 1
        y_min, y_max = data[:, 1].min() - 1, data[:, 1].max() + 1
        step = 0.05
        xx, yy = np.meshgrid(np.arange(x_min, x_max, step), np.arange(y_min, y_max, step))
        Z = self.predict(np.c_[xx.ravel(), yy.ravel()])
        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        # plt.figure()
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        for p,tag in zip(self.data,self.labels): #TRAIN DATA
            if tag:  plt.scatter(p[0],p[1],marker="x",  color="black")
            else:    plt.scatter(p[0],p[1],marker="o",  color="black")

        pred = self.predict(data)
        
        #We change color to red for the TEST data
        for p,tag,pred in zip(data,tags,pred):
            if tag: plt.scatter(p[0],p[1],marker="x",  color="green" if tag==pred else "red")
            else:   plt.scatter(p[0],p[1],marker="o",  color="green" if tag==pred else "red")
        
        plt.show()   # Show the plot

    def getConfusions(self, X, y):
        predictions = self.predict(X)
        TP,TN,FP,FN=0.,0.,0.,0.
        for prediction, target in zip(predictions, y):
            if prediction == 1:
                if target == 1:   TP += 1
                elif target == 0: FP += 1
            elif prediction == 0:
                if target == 1:   FN += 1
                elif target == 0: TN += 1
        #  T F
        #P
        #N  
        return np.matrix([[TP,FP],[FN,TN]])

if __name__ == '__main__':
    X,tags = getDataFromFile("chips.txt")
    x_train, x_test, y_train, y_test = train_test_split(X,tags,test_size=0.1,random_state=0) 
    model = KNN(k=3)
    model.fit(x_train,y_train)
    pred = model.predict(x_test)
    print( f'Accuracy {sum(pred==y_test)}/{len(pred)} {accuracy_score(y_test, pred )}' )
    print(model.getConfusions(x_test,y_test))
    # model.plot(x_test,y_test)

    
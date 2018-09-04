from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from random import random
from matplotlib import cm
import numpy as np
import csv

# Linear regression.
# Gradient descent or genetic algorithm. 
# You can't use existing implementations from libraries. 
# The choice of hyperparameters and the configuration method is left for you, but be prepared to answer additional questions on them. 
# The dataset is the dependence of the cost of housing on the area and the number of rooms. 
# Use mean squared error as an empirical risk.
# Your program must have an ability to get additional input points (e.g., from console) for checking the already trained model.

#Recibe como parametro el path a un archivo csv
#con +2 columnas, "x,x1,x2,...xN,y" dond ela ultima es el valor en Y
def getDataFromFile(filename,headers=False):  
    val = []
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=',')
        if headers: next(reader) #Just skip the headers
        for i in reader:
            val.append( [float(j) for j in i] )
    return np.array(val) #Returns everything as a numpy matrix

#Y are the tags
def meanSquaredError(X,Y,thetas):
    err = 0.0
    for _x,_y in zip(X,Y):
        err += (h(_x,thetas)-_y)**2 #Squared Errors
    return err/(2*len(X))          #mean

def h(thetas,vX): #Hipothesis
    return thetas.transpose().dot( vX )

#s is the size of the returned vector
def getInitialWeights(size,e=0.05): #Freedom
    weights = [0.0]*size
    for i in range(size): weights[i]= -e+(2*e*random()) #Set a value between [-e,+e]    
    return np.array(weights)

def readHouseVectorFromUser():
    while True:
        try:    
            area = float(input("Area: "))
            rooms = int(input("Rooms: "))
            return np.array([area,rooms])
        except ValueError: 
            print("Enter a valid number")
#X Nx2 
#Y Nx1
#f is the predict function, it recieves th0,thetas and a vector of Xs
def plotData(data,tags,thetas,f,xl="X",yl="Y",zl="Z",errors=None,m=1,s=1):
    #Transform information
    X,Y = data[:,0],data[:,1]
    Z   = tags
    #Init
    fig = plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    ax.set_xlabel(xl)
    ax.set_ylabel(yl)
    ax.set_zlabel(zl)
    
    ax.plot(X,Y,Z,linestyle="none",marker="o",mfc="none",markeredgecolor="red")
    # for x,y,z in zip(X,Y,Z):ax.scatter(x,y,z,marker="x") #Scattered
    
    #Regression
    th0,thetas = thetas[0],thetas[1:]
    xx,yy = np.arange(X.min()-1,X.max()+1,0.25),np.arange(Y.min()-1,Y.max()+1,0.25) 
    xx,yy = np.meshgrid(xx,yy)
    R = np.zeros( xx.shape ) #For the resulting plane
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]): 
            R[i][j] = f(th0,thetas,np.array([xx[i][j],yy[i][j]])) #Evaluate Z
    
    ax.plot_surface(xx,yy,R,rstride=2,cstride=2,alpha=0.6,cmap=cm.jet) #jet
    plt.show()

    if not (errors is None):
        plt.plot( errors )
        plt.show()

#Alpha is the step
def gradient(X,Y,thetas=None,test=None,alpha=0.01,iters=1500):
    #Init random thetas if they don't tell us how to start
    if thetas is None: thetas=getInitialWeights(X.shape[1]+1) 
    #Historic of errors for plotting
    errors = np.zeros(iters)                       
    #We need to add a column of 1s for theta0 for better matrix operations
    fixedX = np.append(np.ones((X.shape[0],1)),X,axis=1) 
    
    #For the method not to loop infinitely on diverge
    for it in range(iters): 
        tempThetas = thetas.copy() #Temporals for updates
        for j in range(0, len(tempThetas) ):
            #FORMULA thetaI = thetaI-(alfa/m)*sum( (hip(xi)-yi)*x[j]i  )
            s = sum( ( h(_x,thetas)-_y)*_x[j] for (_x,_y) in zip(fixedX,Y) ) 
            tempThetas[j] = thetas[j] - (alpha/len(X))*s #Moving to the solution
        thetas = tempThetas #Update the thetas
        errors[it]  = meanSquaredError(fixedX,Y,thetas)   #Guardar costos
        #Break if errors < E
    return (errors,thetas)

#Normalizes all, matrix/vector
def normalize(vector,mean=None,sigma=None): 
    mean = vector.mean() if mean is None else mean
    sigma = vector.std() if sigma is None else mean
    #range = vector.max() - vector.min()        #Other type of normalization     
    f = np.vectorize( lambda xi: (xi-mean)/sigma   ) #norm declaration
    return f(vector), mean, sigma                    #eval and return

#vX is the vector of X
def predict(th0,thetas,vX): 
    return  th0+h(thetas,vX)

if __name__ == '__main__':
    print("Read Data: Starting")
    vals = getDataFromFile('prices.txt',headers=True)
    X,Tags = vals[:,:-1] , vals[:,-1] #Split for better handling
    thetas = getInitialWeights(X.shape[1]+1,e=0.001)
    print("Read Data: DONE")

    nX,xMean,xSigma    = normalize(X)
    #Required to be defined here because we need values of xMean and xSigma
    def predict_non_normalized(th0,th,vX): 
        nvx,_,_ = normalize(vX,mean=xMean,sigma=xSigma)
        return predict(th0,th,nvx)

    print("Gradient: Starting")
    errors, thetas     = gradient(nX,Tags,thetas,alpha=0.001,iters=1500)
    th0,fixedThetas = thetas[0],thetas[1:] #For the prediction, better split
    print("Gradient: DONE")
    
    #Added
    # print( errors[-1]**0.5 ) #CurrentError
    # print(thetas)
    # X = np.append(np.ones((nX.shape[0],1)),X,axis=1) 
    # Y = np.reshape(Tags, (47,1))
    # thetas = np.linalg.inv(X.T@(X))@( X.T@( Y ) )  
    # thetas = np.reshape(thetas,(3,))
    # th0F,fixedThetasF = thetas[0],thetas[1:]
    # print( thetas )
    # print( predict(th0F,fixedThetasF, np.array([ 3890,3 ]) ) )

    try:
        c=""    
        while c != "q":
            c = input("Commands:\n\ts\tShow Ploted Data\n\tp\tPredict Value\n\tq\texit\n")
            if c == "s":
                print("Plot: Starting")
                plotData(X,Tags,thetas,predict_non_normalized,xl="Area",yl="Rooms",zl="Price",errors=errors)
                print("Plot: DONE")
            elif c == "p":
                hv = readHouseVectorFromUser()
                houseCostPrediction = predict_non_normalized(th0,fixedThetas,hv)
                print( format(houseCostPrediction, ',.2f') ) #add commas, only 2 decimal numbers
    except KeyboardInterrupt:   
        pass

    print("Bye")


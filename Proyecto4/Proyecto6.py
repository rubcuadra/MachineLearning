import matplotlib.pyplot as plt
import numpy as np
import math, csv, copy
from random import random
from numpy.linalg import inv

#Recibe como parametro el path a un archivo csv
#con +2 columnas, "x,x1,x2,...xN,y" dond ela ultima es el valor en Y
def getDataFromFile(filename):  
    val = [ [],[] ]
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i in reader:
            val[0].append( [float(j) for j in i[0: len(i)-1]])
            val[1].append( float( i[-1] )  )
    return [np.array(val[0]),np.array(val[1])]

#Funcion net, hipothesis
def net(vX,vW):
    return  1 if vW.transpose().dot(vX)>=0.5 else 0

#Funcion escalon de net
def netE( vX, vW): 
    return 1 if net(vX,vW) > 0 else 0 #True or False

#Alpha es el factor ganancia
#Max iters es para que no se quede trabado
#e es la comparacion para considerar un error valido
def entrenaPerceptron(X,Y,weights=None,e=0.001,maxIters=1000,alpha=1):
    return entrenaPerceptron2(**locals())[0] #Solo regresar los pesos que piden

def entrenaAdaline(X,Y,weights=None,e=0.001,alpha=0.001):
    return entrenaAdaline2(**locals())[0] #Solo regresar los pesos que piden
def entrenaAdaline2(X,Y,weights=None,e=0.001,alpha=0.001):
    if weights is None: weights = np.array( [random() for i in range(X.shape[1]+1)] ) #Si no nos dieron pesos vacios usar un random
    elif type(weights).__module__ != np.__name__: weights = np.array(weights)
    X = np.append(np.ones((X.shape[0],1)), X , axis=1) #Appendear 1s izq
    errors = [] #Para graficar, aun no la regresamos
    validError = False
    while not validError:
        l = len(X)
        for i in range(l): #iterar sobre cada ejemplo
            cError = Y[i] - net(X[i],weights)
            for j in range(len(weights)):
                weights[j] += alpha*cError*X[i][j] #ultimo error*Xs en ejemplo actual
            errors.append(cError)
        #Obtener error cuadratico medio aqui
        err = sum(np.array(errors[-l:])**2)/(2*l)
        if err < e: validError = True #Ya acabara esto
    return weights, errors
        
def entrenaPerceptron2(X,Y,weights=None,e=0.001,maxIters=1000,alpha=1):
    if weights is None: weights = np.array( [random() for i in range(X.shape[1]+1)] ) #Si no nos dieron pesos vacios usar un random
    elif type(weights).__module__ != np.__name__: weights = np.array(weights)
    X = np.append(np.ones((X.shape[0],1)), X , axis=1) #Appendear 1s izq
    
    errors = [] #Para graficar, aun no la regresamos
    validError = False
    while not validError:
        validError = True
        for i in range(len(X)): #iterar sobre cada ejemplo
            net = netE(X[i],weights)
            cError = Y[i]-net   #Obtener error actual con valorReal-net
            for j in range(len(weights)):
                weights[j] += alpha*cError*X[i][j] #ultimo error*Xs en ejemplo actual
            #Mientras no sean errores validos seguir iterando (<e)
            if abs(cError) > e: validError = False 
            errors.append(cError) #Agregar al arreglo para graficar
        maxIters-=1
        if maxIters is 0: break #Que no se quede ahi trabado
    return weights, errors

def predicePerceptron(weights,X):
    if type(weights).__module__ != np.__name__: weights = np.array(weights)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if len(X) < len(weights): X = np.append([1],X)  #Agregar 1s a la izq
    return netE(X,weights)

def prediceAdaline(weights,X):
    if type(weights).__module__ != np.__name__: weights = np.array(weights)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if len(X) < len(weights): X = np.append([1],X)  #Agregar 1s a la izq
    return net(X,weights)

#Falta graficar la sigmoidal
#Nos deberian dar una theta normalizada y una X normalizada, Y debe ser 1 o 0
def graficaErrores(errores):
    plt.plot( errores )
    plt.show()   # Show the plot

if __name__ == '__main__':
    fileToUse = "dataOR.csv"
    xData,yData = getDataFromFile(fileToUse)
    pesosPerc,erroresPerc = entrenaPerceptron2(xData,yData)
    pesosAda, erroresAda = entrenaAdaline2(xData,yData)
    print fileToUse
    print "Perceptron :",pesosPerc
    print "Adaline    :",pesosAda

    for x,y in zip(xData,yData): #Validar que es correcta la prediccion
        print x,":",predicePerceptron(pesosPerc,x)    
        print x,":",prediceAdaline(pesosPerc,x)    
        print

    graficaErrores(erroresPerc)
    graficaErrores(erroresAda)
    

    

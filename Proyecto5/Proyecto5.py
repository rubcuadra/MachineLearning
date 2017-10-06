import matplotlib.pyplot as plt
import numpy as np
import math, csv, copy
from random import random
from numpy.linalg import inv
from enum import Enum

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

def normalizarMedia(vector): #Vector con valores de X
    media = vector.mean()
    #rango = vector.max() - vector.min()             
    sigma = vector.std()
    f = np.vectorize( lambda xi: (xi-media)/sigma   ) #Normalizar
    return f(vector)                                  #Evalua cada valor del vector

#Recibe una matrix X y regresa:
#   Una matrix _X normalizada usando la media por columna
#   Un vector mu que contiene las medias de cada columna
#   Un vector sigma que contiene las deviaciones estandares por columna
def normalizacionDeCaracteristicas(X): #VALIDAR SI mu y sigma SON DEL _X o X
    if type(X).__module__ != np.__name__: X = np.array(X)
    _X = np.apply_along_axis( normalizarMedia , 0, X) #Normalizar por cada columna de X
    mu = np.mean(X,axis=0)                            #Vector con medias
    sigma = np.std(X,axis=0)                          #Deviaciones estandares
    return (_X,mu,sigma)

#Pesos
#Numero de neuronas en la capa
#Activacion se toma una del enum
class activaciones(): #(enum) seria lo mejor
    LINEAL = "lineal"
    SIGMOIDAL = "sigmoidal"

#Hipothesis a usar, Vector vX con valores en X y vector thetas
def h(vX,thetas):
    return thetas.transpose().dot(vX)
#La funcion sigmoidal esta dada por 1/(1 + e^(-z))
#Donde z es el resultado de la hipothesis thetaT*x
def sig(z):
    return 1.0/(1.0+np.e**(-z))

#X y Y son vectores de misma size
#Thetas son pesos
def funcionCostoSigmoidal(thetas, X, Y):
    if type(thetas).__module__ != np.__name__: thetas = np.array(thetas)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    #Inicializar outputs
    J = 0
    m = len(X) 
    for i in range(0,m):
        sigEval = sig(h(X[i],thetas))
        pt = -1*Y[i] *np.log(sigEval)
        pf = (1-Y[i])*np.log(1-sigEval)
        J += pt - pf                
    J /= m                              #Multiplicar todo por 1/m, J ya esta
    return J

#Derivada de Costo Sigmoidal 
#g'(z) = g(z)(1 - g(z))
#donde g(z) = sig(z)
def sigmoidGradiente(weights,Xi):
    z  = h(Xi,thetas)
    gz = sig(z)
    return gz*(1-gz)

#Promedio de los errores al cuadrado
def funcionCostoLineal(thetas, X, Y):
    if type(thetas).__module__ != np.__name__: thetas = np.array(thetas)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    
    error = 0
    m = len(X)
    for i in range(m):
        error += (Y[i]-h(X[i],thetas))**2 #Sumar errores cuadrados
    error /= m #Obtener promedio
    return error

#El gradiente para la funcion lineal es 1, g'(z) = d g(z)/dz = 1
#donde g(z) = z; Derivada del costo
def linealGradiante(z):
    return 1

def bpnUnaNeuronaSigmoidal(w,layerSize,X,Y,alpha=0.01,activacion=activaciones.LINEAL, iters=2000):
    if type(w).__module__ != np.__name__: w = np.array(w)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    
    #Inicializar variables
    m  = len(X)                
    b = randInicializaPesos()[0]
    converged = False
    weights = w.copy()

    if activacion is activaciones.LINEAL:
        print "ES LINEAL"
    elif activacion is activaciones.SIGMOIDAL:
        while iters and not converged:
            #Forward
            z = X.dot(weights) + b
            A = sig(z) #Funcion para la sigmoidal, en lineal A = Z
            J = funcionCostoSigmoidal(weights,X,Y) #Evaluar funciones de costo
            #BackPropagation
            dz = A - Y  #Derivada de Funcion Costa en terminos a * derivada a en term z
            dw = X.transpose().dot(dz)/m #dz * X (Promedio /m)
            db = sum(dz)/m
            #Updatear pesos
            weights -= alpha*dw
            b       -= alpha*db
            iters   -= 1
            #Hacer algo para checar convergencia usando J
    #Regresar pesos y la b
    return np.append([b],weights)  
#Inicializa aleatoriamente los pesos de una capa que tienen L_in entradas (unidades de la capa anterior, sin contar el bias). 
#La inicializacion aleatoria se hace para evitar la simetria. Una buena estrategia es generar valores aleatorios en un rango de 
# Este rango garantiza que los parametros se mantienen pequennos y hacen el aprendizaje mas eficiente.
def randInicializaPesos(L_in=1, e=0.12):
    return np.array( [ -e + 2*e*random() for i in range(L_in)] ) 

#Recibe como parametros una matriz de datos X y un vector de pesos,
#ademas una funcion de activacion
def prediceRNYaEntrenada(X,weights,activationFunction):
    X = np.append(np.ones((X.shape[0],1)), X , axis=1) #Agregar columnas de 0s
    return np.apply_along_axis(lambda x:activationFunction(x,weights),1,X)

#Funcion que se pasa como parametro a pediceRNYaEntrenada
#Recibe el vector xi que representa 1 ejemplo de los datos X
#pero este ejemplo siempre tiene un numero 1 en la posicion 0
#util para poder hacer producto punto con el vector weights
def sigmoidalActivation( xi , weights ):
    return 1. if xi.dot(weights)>0.5 else 0.


if __name__ == '__main__':
    fileToUse = "dataOR.csv"
    xData,yData = getDataFromFile(fileToUse)

    nX, mediasX, sigma = normalizacionDeCaracteristicas(xData)  #Normalizar X
    nY, mediasY, sigmaY = normalizacionDeCaracteristicas(yData) #Normalizar Y

    inputs = len(xData[0])                          #Numero de entradas sin contar b para una neurona
    initialWeights = randInicializaPesos(inputs)    #Inicializar pesos para ese numero de entradas

    w = bpnUnaNeuronaSigmoidal(initialWeights,inputs,xData,yData,activacion=activaciones.SIGMOIDAL)
    prediction = prediceRNYaEntrenada(xData,w,sigmoidalActivation)
    
    assert (yData == prediction).all() #Asegurarnos que todas las Y sean iguales a las predicciones
    



    
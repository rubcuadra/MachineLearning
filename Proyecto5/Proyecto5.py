import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
class activaciones(Enum): #seria lo mejor
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
def funcionCostoSigmoidal(thetas,th0,X, Y):
    if type(thetas).__module__ != np.__name__: thetas = np.array(thetas)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    #Inicializar outputs
    J = 0
    m = len(X) 
    for i in range(0,m):
        sigEval = sig( h(X[i],thetas) + th0 )
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
def funcionCostoLineal(thetas,th0,X, Y):
    if type(thetas).__module__ != np.__name__: thetas = np.array(thetas)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    
    error = 0
    m = len(X)
    for i in range(m):
        error += (Y[i]-(th0 + h(X[i],thetas)))**2 #Sumar errores cuadrados
    error /= m #Obtener promedio
    return error

#El gradiente para la funcion lineal es 1, g'(z) = d g(z)/dz = 1
#donde g(z) = z; Derivada del costo
def linealGradiante(thetas,xi):
    return 1

#Las funciones que devuelve son funciones que reciben como parametro:
#   weights : Vector de 1xn (Numero de variables en un ejemplo)
#   b       : Numero 
#   X       : Matrix de mxn
#   Y       : Vector de 1xm (Numero de ejemplos en X)
def getCostFunction(activation):
    if activation is activaciones.LINEAL:
        return funcionCostoLineal
    elif activation is activaciones.SIGMOIDAL:
        return funcionCostoSigmoidal
    else:
        return None

#Regresa una funcion que recibe como parametro un vector Z y devuelve el vector convertido
#(usando sigmoidal, escalon,rELU, etc)
def getAFunction(activation):
    if activation is activaciones.LINEAL:
        return lambda z: z
    elif activation is activaciones.SIGMOIDAL:
        return lambda z: sig(z)
    else:
        return None

#Regresa una funcion que recibe como parametro un 2 vectores de misma size, una A y una Y;
#Regresa un vector con las mismas dimensiones que estos dos
def getDZFunction(activation):
    if activation in [activaciones.LINEAL,activaciones.SIGMOIDAL]:
        return lambda A,Y: A-Y
    else:
        return None

def bpnUnaNeurona(w,layerSize,X,Y,alpha=0.01,e=0.1,activacion=activaciones.LINEAL, iters=1000):
    if type(w).__module__ != np.__name__: w = np.array(w)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    
    #Inicializar variables
    m  = len(X)                
    b = randInicializaPesos()[0]
    costs = []
    converged = False
    weights = w.copy()
    costFunction  = getCostFunction(activacion) #Funcion para calcular J
    A_Function    = getAFunction(activacion)    #Funcion para calcular A
    dz_Function   = getDZFunction(activacion)   #Funcion para obtener dz

    while iters and not converged:
        #Forward
        z = X.dot(weights) + b
        A = A_Function(z) #Funcion para la sigmoidal, en lineal A = Z
        J = costFunction(weights,b,X,Y) #Evaluar funciones de costo
        #BackPropagation
        #dz es la funcion sigmoidal gradiente
        dz = dz_Function(A,Y)  #Derivada de Funcion Costa en terminos a * derivada a en term z
        dw = X.transpose().dot(dz)/m #dz * X (Promedio /m)
        db = sum(dz)/m
        #Updatear pesos
        weights -= alpha*dw
        b       -= alpha*db
        iters   -= 1
        costs.append(J)
        if J < e: #Error permitido
            converged = True
    #Regresar pesos y la b
    return [np.append([b],weights),costs]
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
    #print xi,weights
    return 1. if xi.dot(weights)>0.5 else 0.

def linealActivation( xi , weights ):
    return h( xi,weights )

#Solo sirve cuando X es una matrix de 2xm
def graficaSigDatos(X,Y,theta):
    pX,pY,nX,nY = [],[],[],[]
    for _x,_y in zip(X,Y):
        # plt.scatter( *_x, marker="x" if _y else 'o') #Sin labels jala este
        if _y: pX.append(_x[0]);pY.append(_x[1])
        else:  nX.append(_x[0]);nY.append(_x[1])
    plt.scatter(pX,pY, marker="x", label="True" )
    plt.scatter(nX,nY, marker="o", label="False" )
    #Hacer la linea    
    x1_min = np.amin(X,axis=0)[0] -1
    x1_max = np.amax(X,axis=0)[0] +1
    #Dos valores es suficiente puesto que es un recta
    xs = [x1_min, x1_max]
    #0.5 es la brecha de cuando pasa o no pasa el examen
    f = lambda x1,th : (0.5-th[0]-th[1]*x1)/th[2]
    #Evaluar x2 por cada x1
    plt.plot( xs  , [f(xi,theta) for xi in xs] )
    plt.legend() # Add a legend
    plt.show()   # Show the plot

def graficarCostos(Js):
    plt.plot( Js, label='Historial del error')
    plt.legend() # Add a legend
    plt.show()   # Show the plot

if __name__ == '__main__':
    sigmoidal = True #False -> Lineal
    if sigmoidal:
        fileToUse = "dataAND.csv"
        xData,yData = getDataFromFile(fileToUse)
        inputs = len(xData[0])                          #Numero de entradas sin contar b para una neurona
        initialWeights = randInicializaPesos(inputs)    #Inicializar pesos para ese numero de entradas
        w,costos = bpnUnaNeurona(initialWeights,inputs,xData,yData,alpha=0.1,iters=2000,activacion=activaciones.SIGMOIDAL)
        prediction = prediceRNYaEntrenada(xData,w,sigmoidalActivation)
        graficaSigDatos(xData,yData,w)
    else: #Lineal
        fileToUse = "dataCasas.csv"
        xData,yData = getDataFromFile(fileToUse)
        nX, mediasX, sigma = normalizacionDeCaracteristicas(xData)  #Normalizar X
        nY, mediasY, sigmaY = normalizacionDeCaracteristicas(yData) #Normalizar Y
        inputs = len(xData[0])                          #Numero de entradas sin contar b para una neurona
        initialWeights = randInicializaPesos(inputs)    #Inicializar pesos para ese numero de entradas
        w,costos = bpnUnaNeurona(initialWeights,inputs,nX,yData,alpha=0.001,iters=5000,activacion=activaciones.LINEAL)
        prediction = prediceRNYaEntrenada(nX,w,linealActivation)
        graficarCostos(costos)
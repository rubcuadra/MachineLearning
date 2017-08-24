import matplotlib.pyplot as plt
import numpy as np
import math, csv, copy
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

def normalizarMedia(vector): #Vector con valores de X
    media = vector.mean()
    rango = vector.max() - vector.min()             
    f = np.vectorize( lambda xi: (xi-media)/rango   ) #Normalizar
    return f(vector)      #Evalua cada valor del vector

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

# thetas = ((Xt*X)-1) *Xt*y
def ecuacionNormal(X,Y):
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(X)
    fixedX = np.append(np.ones((X.shape[0],1)), X , axis=1) #Agregar ones a X
    return  (inv(fixedX.transpose().dot(fixedX))).dot(fixedX.transpose().dot(Y))

def calculaCosto(x,y,theta):
    r = 0
    for (_x,_y) in zip(x,y):
        r += (hipothesis(_x,theta)-_y)**2
    return r/(2*len(x))

def graficaError(historial):
    plt.plot( historial, label='Historial del error')
    plt.legend() # Add a legend
    plt.show()   # Show the plot

def hipothesis(vX,thetas):
    return thetas.transpose().dot(vX)

def gradienteDescendenteMultivariable(X,Y,theta=None,alpha=0.01,iteraciones=1500):
    '''Validar que todo sea Numpy array/matrix'''
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    if theta is None: theta = np.zeros(X.shape[1]+1)  #Si no nos dieron theta llenarla con numero de Xs+1
    elif type(theta).__module__ != np.__name__: theta = np.array(theta)

    #Comenzar a trabajar
    jHistoria = np.zeros(iteraciones)                       #Historial de iteraciones
    fixedX = np.append(np.ones((X.shape[0],1)), X , axis=1) #Agregar ones a X
    
    for it in range(0,iteraciones): 
        tempThetas = theta.copy() #Thetas temporales
        for j in range(0, len(tempThetas) ):
            #FORMULA thetaI = thetaI-(alfa/m)*sum( (hip(xi)-yi)*x[j]i  )
            tempThetas[j] = theta[j] - (alpha/len(X))*sum( ( hipothesis(_x,theta)-_y)*_x[j] for (_x,_y) in zip(fixedX,Y) )
        theta = tempThetas
        jHistoria[it]  = calculaCosto(fixedX,Y,theta)   #Guardar costos
    return (jHistoria,theta)

def predicePrecio(X,theta):
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(theta).__module__ != np.__name__: theta = np.array(theta)
    return hipothesis( np.append([1],X) ,theta)

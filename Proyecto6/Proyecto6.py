#Ruben Cuadra A01019102
#import gnumpy as gpu
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math, csv, copy
from random import random
from enum import Enum
import json

#Activacion se toma una del enum
class activaciones(Enum): #seria lo mejor
    LINEAL = "lineal"
    SIGMOIDAL = "sigmoidal"

#Activacion viene del enum
#size es un numero entero que representa las neuronas en la capa
class NNLayer(object):
    def __init__(self,numNeuronas,activacion):
        super(NNLayer,self).__init__()
        self.size = numNeuronas
        self.activacion = activacion

def reverse_enum(L):
    for index in reversed(xrange(len(L))):
        yield index, L[index]

#El formato es archivos donde cada linea contiene multiples numeros (entre -1 y 1) separados por un espacio, el ultimo numero representa una etiqueta para esos valores
#Ej. "1 0 0 0 -1 0 0 -0.12312 0.21321321 0 6", donde 6 seria la etiqueta para esos valores
def getDataFromFile(filename,limit=float("inf"),delimiter=" "):  
    val = [ [],[] ]
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for c,i in enumerate(reader):
            if c >= limit: break #No cargar todos los numeros de golpe
            if i: #Caracteres ''
                val[0].append( [float(j) for j in i[0: len(i)-1]])
                val[1].append( float( i[-1] )  )
    return [np.array(val[0]),np.array(val[1])]

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
def sigmoidGradiente(z):
    gz = sig(z)
    return gz*(1-gz)

#Promedio de los errores al cuadrado
def funcionCostoLineal(thetas,th0,X, Y):
    error = 0
    m = len(X)
    for i in range(m):
        error += (Y[i]-(th0 + h(X[i],thetas)))**2 #Sumar errores cuadrados
    error /= m #Obtener promedio
    return error

#El gradiente para la funcion lineal es 1, g'(z) = d g(z)/dz = 1
#donde g(z) = z; Derivada del costo
def linealGradiante(z):
    return 1

#Devuelve funciones de gradientes
def getdGFunction(activation):
    if activation is activaciones.LINEAL:
        return linealGradiante
    elif activation is activaciones.SIGMOIDAL:
        return sigmoidGradiente
    else:
        return None

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
def getActivationFunction(activation):
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

def getYsAsMatrix(Y,totalLabels):
    #Identidad con todas las etiqueras posibles
    t = np.identity(totalLabels) 
    f = np.vectorize(lambda y: t[y], otypes=[np.ndarray])
    #Poner otypes para que nos deje
    return f(Y)

#A debe tener las evaluaciones sigmoidales por neurona
def getCost(A,Y): 
    J = 0
    m = Y.shape[0]
    #sx es la evaluacion sigmoidal en cada ejemplo
    #su Size sera numEjemplosXnumNeuronasSalida
    for a,y in zip(A.T,Y): #Iterar por cada ejemplo
        pt = -1*y*np.log(a)
        pf = (1-y)*np.log(1-a)
        J += pt-pf  
    J /= m
    return np.average(J)

#input_layer_size representa la cantidad de entradas para cada ejemplo, en una imagen de 20x20 tendria un size de 400
#hidden_layer_size representa la cantidad de neuronas en la capa de enmedio(Deberia ser un vector si queremos N capas)
#num_labels es la cantidad de salidas en la red, cada salida representa un posible grupo al que pertenece el ejemplo, para detectar digitos existen 10 salidas 0-9
#y el valor de las etiquetas
#X cada valor posee un ejemplo
def entrenaRN(X,Y,hidden_layers,iters=1000,alpha=0.001,activacionFinal=activaciones.SIGMOIDAL):
    input_layer_size = len( X[0] )
    num_labels       = len( np.unique(Y) )
    total_layers     = len( hidden_layers )
    fixedYs          = getYsAsMatrix(Y,num_labels)  
    #Y                = fixedYs
    p                = {"A0":X.T} #Para iterar despues
    m                = p["A0"].shape[1]
    # print "A0",p["A0"].shape
    # print "Y" ,Y.shape
    finalLayer = NNLayer(num_labels,activacionFinal) 
    layers = hidden_layers+[finalLayer]
    #Generar pesos aleatorios iniciales, pasar a una funcion que nos devuelva el dictionary
    #+[num_labels] donde num_labels es la cantidad de neuronas en la capa final(Categorias para clasificar)
    l_in = input_layer_size
    for i,layer in enumerate(layers): #Iterar sobre cada capa
        i+=1
        p["W%s"%i] = randInicializacionPesos(l_in,layer.size)
        p["b%s"%i] = randInicializacionPesos(1,layer.size)   #Obtener solo la b para cada neurona
        l_in = layer.size #Preparar la size de la capa anterior   

    #AQUI EMPIEZA LA ITERACION
    for i in xrange(iters):    
        #Iterar por cada capa, de momento la getActivationFunction sera igual entre todas las neuronas de esa capa
        #FORWARD PROPAGATION
        for l,layer in enumerate(layers):
            A_Function    = getActivationFunction(layer.activacion)    #Funcion para calcular A
            Zi, Wi, bi    = "Z%s"%(l+1), "W%s"%(l+1), "b%s"%(l+1)
            Ai, Ap  = "A%s"%(l+1),"A%s"%(l)
            p[Zi]   = p[Wi].dot( p[Ap] ) + p[bi]
            p[Ai]   = A_Function(  p[Zi]  ) 
        
        #BACK PROPAGATION
        l = total_layers+1 #Capa Final
        dz_Function      = getDZFunction(finalLayer.activacion)
        dZi, dWi, dbi    = "dZ%s"%l, "dW%s"%l,"db%s"%l
        p[dZi] = dz_Function(p[Ai],Y) #Con vector este tarda demasiado
        p[dWi] = p[dZi].dot(p[Ap].T)
        p[dbi] = np.sum(p[dZi],axis=1,keepdims=True)/m

        for l,layer in reverse_enum(hidden_layers): #DEMAS CAPAS
            l += 1
            activacion    = layer.activacion 
            dz_Function   = getDZFunction(activacion)
            dg_function   = getdGFunction(activacion)
            dZi, dWi, dbi = "dZ%s"%l, "dW%s"%l,"db%s"%l
            dZn, dWn, Wn  = "dZ%s"%(l+1), "dW%s"%(l+1), "W%s"%(l+1)
            Zi , bi, Wi   = "Z%s"%l,"b%s"%l, "W%s"%l
            Ap            = "A%s"%(l-1)
            p[dZi] = p[Wn].T.dot( p[dZn] ) * dg_function(p[Zi])#*g'(Z)
            p[dbi] = np.sum(p[dZi],axis=1,keepdims=True)
            p[dWi] = p[dZi].dot( p[Ap].T )        
            # #Ajustar pesos    
            p[Wi]  = p[Wi] - p[dWi]*alpha/m
            p[bi]  = p[bi] - p[dbi]*alpha/m  

        #Obtener Costo
        J = getCost( p[Ai], Y)
        print J
    return p #p debe tener todas las Ws

#Para una capa
#   L_in  seria la size del vector Wi
#   L_out seria la cantidad de neuronas en la capa, valor maximo de i en Wi
def randInicializacionPesos(L_in,L_out,e=0.12):
    weights = []
    for _ in range(L_out):
        weights.append( np.array( [ -e + 2*e*random() for i in range(L_in)] )  )
    return np.array(weights)

#X son todos los ejemplos
#W es un vector donde posee W[0] = W0, W[1] = W1...
#b es un vector donde posee b[0] = b0, b[1] = b1...
def prediceRNYaEntrenada(X,W,b):
    pass

if __name__ == '__main__':
    xExamples,tags = getDataFromFile("digitos.txt")
    #print xExamples,tags
    l  = NNLayer(25,activaciones.LINEAL)
    p  = entrenaRN(xExamples,tags,[l],iters=1,alpha=0.01)
    np.save('network.npy',p) 
    #p = np.load('network.npy').item()

#Ruben Cuadra A01019102
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import math, csv, copy
from random import random
from enum import Enum

#Activacion se toma una del enum
class activaciones(Enum): #seria lo mejor
    LINEAL = "lineal"
    SIGMOIDAL = "sigmoidal"

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
def sigmoidGradiente(z):
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
def linealGradiante(z):
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

def getCosto(Y): 
    #k posibles etiqueras 
    print Y.shape
    #pt = -1*Y[i] *np.log(sigEval)
    #pf = (1-Y[i])*np.log(1-sigEval)
    return 0

def getYsAsMatrix(totalLabels):
    # m    = []
    # for i in range(totalLabels):
    #     t = np.zeros( totalLabels )
    #     t[i] = 1. #Solo nos dice a que grupo pertenece
    #     m.append( t )
    return np.identity(totalLabels)

#input_layer_size representa la cantidad de entradas para cada ejemplo, en una imagen de 20x20 tendria un size de 400
#hidden_layer_size representa la cantidad de neuronas en la capa de enmedio(Deberia ser un vector si queremos N capas)
#num_labels es la cantidad de salidas en la red, cada salida representa un posible grupo al que pertenece el ejemplo, para detectar digitos existen 10 salidas 0-9
#y el valor de las etiquetas
#X cada valor posee un ejemplo
def entrenaRN(X,Y,hidden_layers_sizes,iters=1000,alpha=0.001):
    input_layer_size = len( X[0] )
    num_labels       = len( np.unique(Y) )
    total_layers     = len( hidden_layers_sizes )
    fixedYs          = getYsAsMatrix(num_labels)  
    p                = {"A0":X.T} #Para iterar despues
    m                = p["A0"].shape[1]
    # print "A0",p["A0"].shape
    # print "Y" ,Y.shape
    
    #Generar pesos aleatorios iniciales, pasar a una funcion que nos devuelva el dictionary
    #+[num_labels] donde num_labels es la cantidad de neuronas en la capa final(Categorias para clasificar)
    l_in = input_layer_size
    for i,layer_size in enumerate(hidden_layers_sizes+[num_labels]): #Iterar sobre cada capa
        i+=1
        p["W%s"%i] = randInicializacionPesos(l_in,layer_size)
        p["b%s"%i] = randInicializacionPesos(1,layer_size)   #Obtener solo la b para cada neurona
        l_in = layer_size   

    #AQUI EMPIEZA LA ITERACION
    for i in xrange(iters):    
        #Iterar por cada capa, de momento la getAFunction sera igual entre todas las neuronas de esa capa
        #FORWARD PROPAGATION
        for i,layer in enumerate(hidden_layers_sizes):
            activacion    = activaciones.LINEAL         #Obtenerla por cada capa, remplazar el _ del iterador
            A_Function    = getAFunction(activacion)    #Funcion para calcular A
            Zi, Wi, bi    = "Z%s"%(i+1), "W%s"%(i+1), "b%s"%(i+1)
            Ai, Ap  = "A%s"%(i+1),"A%s"%(i)
            p[Zi]   = p[Wi].dot( p[Ap] ) + p[bi]
            p[Ai]   = A_Function(  p[Zi]  )
            # print Wi,p[Wi].shape
            # print bi,p[bi].shape
            # print Ai,p[Ai].shape
            # print Zi,p[Zi].shape
        #Trabajar sobre la capa final
        i = total_layers+1                      
        #FORWARD Capa Final - Sigmoidal
        activacionFinal = activaciones.SIGMOIDAL
        A_Function    = getAFunction(activacion)    #Funcion para calcular A
        Zi, Wi, bi    = "Z%s"%i, "W%s"%i,"b%s"%i
        Ai, Ap  = "A%s"%i,"A%s"%(i-1)
        p[Zi]   = p[Wi].dot( p[Ap] ) + p[bi] #Por que no se tuvo que hacer T ??
        p[Ai]   = A_Function(  p[Zi]  )
        # print Wi,p[Wi].shape
        # print bi,p[bi].shape
        # print Ai,p[Ai].shape
        # print Zi,p[Zi].shape
        
        J = getCosto( fixedYs )
        
        #Backward Capa Final - Sigmoidal
        dz_Function   = getDZFunction(activacion)
        dZi, dWi, dbi    = "dZ%s"%i, "dW%s"%i,"db%s"%i
        p[dZi] = p[Ai] - Y #Checar lo de las Ys y el vector de 10 posiciones
        p[dWi] = p[dZi].dot(p[Ap].T)
        p[dbi] = np.sum(p[dZi],axis=1,keepdims=True)/m
        # print dZi,p[dZi].shape
        # print dWi,p[dWi].shape
        # print dbi,p[dbi].shape

        #Backward demas Capas
        for i,layer in reverse_enum(hidden_layers_sizes):
            i += 1
            activacion    = activaciones.LINEAL         #Obtenerla por cada capa, remplazar el _ del iterador
            dz_Function   = getDZFunction(activacion)
            dZi, dWi, dbi = "dZ%s"%i, "dW%s"%i,"db%s"%i
            dZn, dWn      = "dZ%s"%(i+1), "dW%s"%(i+1)
            Zi , Ap, Wi   = "Z%s"%i,"A%s"%(i-1), "W%s"%i
            bi = "b%s"%i
            Wn = "W%s"%(i+1)
            p[dZi] = p[Wn].T.dot( p[dZn] ) # * g'( p[Zi] ) g' se debe obtener de otra funcion
            p[dbi] = np.sum(p[dZi],axis=1,keepdims=True)
            p[dWi] = p[dZi].dot( p[Ap].T )        
            # #Ajustar pesos    
            p[Wi]  = p[Wi] - p[dWi]*alpha/m
            p[bi]  = p[bi] - p[dbi]*alpha/m
            
    
#Para una capa
#   L_in  seria la size del vector Wi
#   L_out seria la cantidad de neuronas en la capa, valor maximo de i en Wi
def randInicializacionPesos(L_in,L_out,e=0.12):
    weights = []
    for _ in range(L_out):
        weights.append( np.array( [ -e + 2*e*random() for i in range(L_in)] )  )
    return np.array(weights)

if __name__ == '__main__':
    xExamples,tags = getDataFromFile("digitos.txt")
    #print xExamples,tags
    entrenaRN(xExamples,tags,[25],iters=1)

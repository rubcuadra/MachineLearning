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

#El formato es archivos donde cada linea contiene multiples numeros (entre -1 y 1) separados por un espacio, el ultimo numero representa una etiqueta para esos valores
#Ej. "1 0 0 0 -1 0 0 -0.12312 0.21321321 0 6", donde 6 seria la etiqueta para esos valores
def getDataFromFile(filename,limit=float("inf"),delimiter=" "):  
    val = [ [],[] ]
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for c,i in enumerate(reader):
            if c >= limit: break #No cargar todos los numeros de golpe
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

#input_layer_size representa la cantidad de entradas para cada ejemplo, en una imagen de 20x20 tendria un size de 400
#hidden_layer_size representa la cantidad de neuronas en la capa de enmedio(Deberia ser un vector si queremos N capas)
#num_labels es la cantidad de salidas en la red, cada salida representa un posible grupo al que pertenece el ejemplo, para detectar digitos existen 10 salidas 0-9
#y el valor de las etiquetas
#X cada valor posee un ejemplo
def entrenaRN(X,y,hidden_layers_sizes):
    input_layer_size = len( X[0] )
    num_labels       = len( np.unique(y) )
    total_layers     = len( hidden_layers_sizes )
    p                = {"A0":X} #Para iterar despues
    
    #Generar pesos aleatorios iniciales, pasar a una funcion que nos devuelva el dictionary
    #+[num_labels] donde num_labels es la cantidad de neuronas en la capa final(Categorias para clasificar)
    l_in = input_layer_size
    for i,layer_size in enumerate(hidden_layers_sizes+[num_labels]): #Iterar sobre cada capa
        i+=1
        p["W%s"%i] = randInicializacionPesos(l_in,layer_size)
        p["b%s"%i] = randInicializacionPesos(1,layer_size)   #Obtener solo la b para cada neurona
        l_in = layer_size        

    #Iterar por cada capa, de momento la getAFunction sera igual entre todas las neuronas de esa capa
    #FORWARD PROPAGATION
    for i,_ in enumerate(hidden_layers_sizes):
        activacion    = activaciones.LINEAL         #Obtenerla por cada capa, remplazar el _ del iterador
        A_Function    = getAFunction(activacion)    #Funcion para calcular A
        dz_Function   = getDZFunction(activacion)   #Funcion para obtener dz
        Zi, Wi, bi    = "Z%s"%(i+1), "W%s"%(i+1), "b%s"%(i+1)
        Ai, Ap  = "A%s"%(i+1),"A%s"%(i)
        p[Zi]   = p[Wi].dot( p[Ap].T) + p[bi]
        p[Ai]   = A_Function(  p[Zi]  )
    
    #Falta FORWARD para la capa final, esa es sigmoidal. W%s % total_layers+1
    activacionFinal = activaciones.SIGMOIDAL
    A_Function    = getAFunction(activacion)    #Funcion para calcular A
    dz_Function   = getDZFunction(activacion)   #Funcion para obtener dz
    i = total_layers+1                          #Poner la capa final
    Zi, Wi, bi    = "Z%s"%(i+1), "W%s"%(i+1), "b%s"%(i+1)
    Ai, Ap  = "A%s"%(i+1),"A%s"%(i)
    p[Zi]   = p[Wi].dot( p[Ap].T) + p[bi]
    p[Ai]   = A_Function(  p[Zi]  )

    for key in p:
        print key

    #print p["A1"]



#Para una capa
#   L_in  seria la size del vector Wi
#   L_out seria la cantidad de neuronas en la capa, valor maximo de i en Wi
def randInicializacionPesos(L_in,L_out,e=0.12):
    weights = []
    for _ in range(L_out):
        weights.append( np.array( [ -e + 2*e*random() for i in range(L_in)] )  )
    return np.array(weights)

if __name__ == '__main__':
    xExamples,tags = getDataFromFile("digitos.txt",limit=10)
    #print xExamples,tags
    entrenaRN(xExamples,tags,[25])

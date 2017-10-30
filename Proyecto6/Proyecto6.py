#Ruben Cuadra A01019102
#import gnumpy as gpu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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
    #Poner otypes para que nos deje
    f = np.vectorize(lambda y: t[y], otypes=[np.ndarray])
    #np.vstack nos convierte de un vector de vectores a una np matrix
    return np.vstack( f(Y) ) #f(Y) convierte los valores en vectores

#A debe tener las evaluaciones sigmoidales por neurona
def getCost(A,Y): 
    J = 0
    m = Y.shape[0]
    #a es la evaluacion sigmoidal para cada ejemplo
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
def entrenaRN(X,Y,hidden_layers=[],iters=1000,e=0.001,alpha=0.001,activacionFinal=activaciones.SIGMOIDAL):
    input_layer_size = len( X[0] )
    num_labels       = len( np.unique(Y) )
    total_layers     = len( hidden_layers )
    fixedYs          = getYsAsMatrix(Y,num_labels)  
    Y                = fixedYs
    p                = {"A0":X.T} #Para iterar despues
    m                = p["A0"].shape[1]
    converged        = False
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
    pJ = float("-inf")
    #AQUI EMPIEZA LA ITERACION
    while (not converged) and (iters>0):
        iters-=1
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
        Wi,bi            = "W%s"%l,"b%s"%l
        Ai, Ap           = "A%s"%l,"A%s"%(l-1)
        p[dZi] = dz_Function(p[Ai], Y.T) #Con vector este tarda demasiado
        
        #ESTO SE MODIFICO, CHECAR QUE SHOW
        dWi = p[dZi].dot(p[Ap].T)
        dbi = np.sum(p[dZi],axis=1,keepdims=True)/m
        p[Wi]  = p[Wi] - dWi*alpha/m
        p[bi]  = p[bi] - dbi*alpha/m  

        for l,layer in reverse_enum(hidden_layers): #DEMAS CAPAS
            l += 1
            activacion    = layer.activacion 
            dz_Function   = getDZFunction(activacion)
            dg_function   = getdGFunction(activacion)
            dZi,dZn, Wn   = "dZ%s"%l,"dZ%s"%(l+1), "W%s"%(l+1)
            Zi , bi, Wi   = "Z%s"%l,"b%s"%l, "W%s"%l
            Ap            = "A%s"%(l-1)
            p[dZi] = p[Wn].T.dot( p[dZn] ) * dg_function(p[Zi])#*g'(Z)
            dbi    = np.sum(p[dZi],axis=1,keepdims=True)
            dWi    = p[dZi].dot( p[Ap].T )        
            #Ajustar pesos    
            p[Wi]  = p[Wi] - dWi*alpha/m
            p[bi]  = p[bi] - dbi*alpha/m  

        #Obtener Costo
        J = getCost( p[Ai], Y) 
        if J < e: converged=True
        #Fix divergencia
        if J > pJ: alpha*=0.9 
        pJ = J                
        print (J)
    p["l"] = len(layers) #Necesario para la prediccion, numero de capas totales
    return p #maybe sacar todas las dZ del dict

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
    #Convertir de shape (n,) -> (n,1)
    if len(X.shape) < 2: X = X[:,np.newaxis].T
    #De momento todas seran activaciones normales y la ultima sigmoidal
    Ap = X.T
    for i,(Wi,bi) in enumerate(zip(W,b)):   
        Zi  = Wi.dot(Ap) + bi
        Ap  = Zi #Aqui se debe aplicar la funcion por capa, al ser lineal es igual    
    Ai = sig(Ap)
    #Obtener el index del elemento mas grande por cada row en Ai
    r = np.argmax(Ai, axis=0) #Cada indice representa el digito reconocido
    #No es necesario parsear el valor maximo
    return r


def getWeightsFromFile(filename):
    d = np.load(filename).item() #dict de la red entrenada
    l = d["l"] #Numero de capas

    #Inicializar retornos
    W = [None]*l
    b = [None]*l

    for key in d:
        #Validar las primeras letas
        #-1 necesario por que las capas empiezan en 1
        if key[0]   is "W": 
            i    = int( key[1:] ) -1
            W[i] = d[key]
        elif key[0] is "b":
            i    = int( key[1:] ) -1
            b[i] = d[key]
    return W, b

#Y son los valores reales de los datos
#_Y es una prediccion
#Nos devuelve el % de error entre ambos arreglos
def getErrorPercentage(Y,_Y):
    e,t = 0.,len(_Y)
    for  (_y,y) in zip(_Y,Y):
        if _y != y:
            e+=1
    #print "%s errores de %s"%(e,t)
    return (e/t)

def graficarNumero( num ):
    s = num.shape[0]**0.5 #Solo graficara numeros de dimensiones width = height
    f = np.vectorize( lambda val: (val+1)/2 ) #Del campo [-1,1] -> [0,1]
    num = f(num)                              #Convertir al campo de 0 a 1
    img = num.reshape( (s,s) )                
    plt.imshow(img.T, interpolation='nearest', cmap='gray') #Escala de grises de 0 a 1
    plt.show()

if __name__ == '__main__':
    xExamples,tags = getDataFromFile("digitos.txt")
    print ("X", xExamples.shape)
    print ("Y", tags.shape)
    entrenar = False
    if entrenar:
        l   = [ NNLayer(25,activaciones.LINEAL) ]
        p   = entrenaRN(xExamples,tags,l,iters=30000,alpha=0.15,e=0.02)
        np.save('network.npy',p) 
    else:
        W,b = getWeightsFromFile("network038_27.npy")
        _Y  = prediceRNYaEntrenada(xExamples,W,b)
        error = getErrorPercentage(tags,_Y)
        print ("%s%% de exito"%(100-error*100))
        #Graficar algo
        example = 3014 #Menor a 5000
        x,tag = xExamples[example],[tags[example]] #tag debe ser un array
        _Y  = prediceRNYaEntrenada(x,W,b)          #_Y es un array
        error = getErrorPercentage(tag,_Y)         #Recibe 2 arrays
        print ("Prediccion: %s"%_Y)
        print ("Etiqueta  : %s"%tag)
        graficarNumero( x )

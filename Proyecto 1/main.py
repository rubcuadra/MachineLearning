import matplotlib.pyplot as plt
import numpy as np
import math
import csv
#Recibe como parametro el path a un archivo csv
#con 2 columnas, "x,y"
def getDataFromFile(filename):  
    val = [ [],[] ]
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=',')
        for i in reader:
            val[0].append( [float(j) for j in i[0: len(i)-1]])
            val[1].append( float(i[1])  )
    return val

def getThetas(x,y):
    m = len(x)                      #Cantidad de datos
    Ex = sum(i for i in x)          #Sumatoria de datos en X
    ExCuad = sum(i*i for i in x)    #Sumatoria de datos^2 en X
    Ey = sum(i for i in y)          #Sumatoria de datos en Y

    Exy = 0
    for (_x,_y) in zip(x,y):        #Suma de datos Xi*Yi
        Exy += _x*_y 
        
    A = np.array([[m,Ex],           #Matrix con [   m  , Sum(X) ]
              [Ex,ExCuad]])         #           [Sum(X), Sum(X^2)]
    Y = np.array([Ey                #Vector con [ Sum(Y)
                ,Exy])              #             Sum(XY)]
    
    Ainv = np.linalg.inv(A)         #Inversa de Matrix A
    return Ainv.dot(Y)              #A-1 * Y nos dara thetas

#Recibe un arreglo de _x(N posiciones, donde la primera posicion es 1s) y uno de _thetas(N posiciones)
def h_t(_x,_thetas):
    result = 0
    for (x_,t_) in zip(_x,_thetas):
        result+= x_*t_ 
    return result #Calcular Y usando thetas

def graficarDatos(x,y,_thetas):
    #plt.scatter(x,y) #Puntos X/Y
    plt.plot(x, [ h_t(i, _thetas) for i in x], label='Regresion')
    plt.legend() # Add a legend
    plt.show()   # Show the plot

#x es un arreglo de 1 a N x's
def gradienteDescendente(x,y,hipothesis,theta,alpha=0.01,iteraciones=1500):
    tempThetas = theta[:] #Copiar thetas a un arreglo temporal
    for i in range(0,iteraciones):  #Dependiendo cuantas veces nos pidan iterar
        for j in range(0, len(tempThetas) ):  #Generar por cada theta
            
            s = sum( (hipothesis(_x,theta)-_y)*_x[j] for (_x,_y) in zip(x,y) )

            tempThetas[j] = theta[j] - (alpha/len(x))*s
        theta = tempThetas[:]
    return theta

def calculaCosto(x,y,theta):
    return sum( math.pow( (h_t(_x,theta)-_y),2 )  for (_x,_y) in zip(x,y) )/(2*len(x))

xData, yData = getDataFromFile('datos.csv')
for i in range(0,len(xData)): xData[i].insert(0,1.0) #Agregar 1s al inicio

if False: #If true es con matrix
    thetas = getThetas(xData,yData)
    calculaCosto(xData,yData,thetas)
    graficarDatos(xData,yData,thetas)
else:     #Else gradiente descendente
    thetas = gradienteDescendente(xData, yData,h_t, [0]*len(xData[0]) ) 
    costo = calculaCosto(xData,yData,thetas)
    print ('Theta 0: %s\nTheta 1: %s'%(thetas[0] ,thetas[1]))
    print ('Resultado funcion de costo: %s\n'%costo)
    graficarDatos(xData,yData,thetas)

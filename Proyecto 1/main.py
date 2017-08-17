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
            val[0].append( float(i[0])  )
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

def h_t(_x,_thetas):
    return _thetas[0] + _thetas[1]*_x #Calcular Y usando thetas

def graficarDatos(x,y,_thetas):
    plt.scatter(x,y) #Puntos X/Y
    plt.plot(x, [ h_t(i, _thetas) for i in x], label='Regresion')
    plt.legend() # Add a legend
    plt.show()   # Show the plot

def gradienteDescendente(x,y,theta=[0,0],alpha=0.01,iteraciones=1500):
    for i in range(0,iteraciones):
        theta0 = theta[0] - (alpha/len(x))* sum( h_t(_x,theta)-_y for (_x,_y) in zip(x,y) )
        theta1 = theta[1] - (alpha/len(x))* sum( (h_t(_x,theta)-_y)*_x for (_x,_y) in zip(x,y) )        
        theta[0] = theta0
        theta[1] = theta1
    return theta

def calculaCosto(x,y,theta):
    return sum( math.pow( (h_t(_x,theta)-_y),2 )  for (_x,_y) in zip(x,y) )/(2*len(x))

xData, yData = getDataFromFile('ex1data1.csv')

if False: #If true es con matrix
    thetas = getThetas(xData,yData)
    calculaCosto(xData,yData,thetas)
    graficarDatos(xData,yData,thetas)
else:     #Else gradiente descendente
    thetas = gradienteDescendente(xData,yData)
    calculaCosto(xData,yData,thetas)
    graficarDatos(xData,yData,thetas)

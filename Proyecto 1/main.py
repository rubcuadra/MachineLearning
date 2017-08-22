import matplotlib.pyplot as plt
import numpy as np
import math, csv, copy
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
#Hacer que no considere el primer valor de las xs?
def graficarDatos(x,y,_thetas):
    for _x,_y in zip(x,y):
        for xi in _x:
            plt.scatter(xi,_y) #Puntos X/Y
    plt.plot(x, [ h_t(i, _thetas) for i in x], label='Regresion')
    plt.legend() # Add a legend
    plt.show()   # Show the plot

#x es una matriz de N*Y elementos, la primer columna esta llena de 1s
#y es un arreglo de N elementos
#ht es una hipothesis, una funcion que recibe dos arreglos de misma size
#theta es un arreglo de Y elementos
def gradienteDescendente(x,y,ht,theta,alpha=0.01,iteraciones=1500):
    tempThetas = theta[:] #Copiar thetas a un arreglo temporal
    for i in range(0,iteraciones):  #Dependiendo cuantas veces nos pidan iterar
        for j in range(0, len(tempThetas) ):  #Generar por cada theta
            s = sum( (ht(_x,theta)-_y)*_x[j] for (_x,_y) in zip(x,y) )
            tempThetas[j] = theta[j] - (alpha/len(x))*s
        theta = tempThetas[:]
    return theta

#Recibe la matriz x, un vector Y, una funcion que sera la hipothesis y un vector theta
def calculaCosto(x,y,ht,theta):
    return sum( math.pow( (ht(_x,theta)-_y),2 )  for (_x,_y) in zip(x,y) )/(2*len(x))

xData, yData = getDataFromFile('datos.csv')
#Definir hipothesis, la cual sera una funcion que recibe 2 arreglos de misma size
#el primero representa un vector X y el otro uno theta
hipothesis = h_t
#Crear vector thetas vacio con size = al numero de variables Xs que tendremos(size de row de xData +1)
thetas = [0]*(len(xData[0])+1)
#Agregar 1s al inicio al xData
xDataCon1s = xData[:]
for i in range(0,len(xDataCon1s)): xDataCon1s[i] = [1] + xDataCon1s[i]
#Calcular thetas usando gradiente Desc, h_t es nuestra hipotesis 
#para que funcione esa h_t es necesario agregarle los 1s al inicio a los(paso anterior)
thetas = gradienteDescendente(xDataCon1s, yData, hipothesis, thetas ) 
costo = calculaCosto(xDataCon1s,yData, hipothesis,thetas)
#Imprimir resultados
for t in range(len(thetas)): 
    print ('Theta %s: %s'%(t,thetas[t])  )
print ('Resultado funcion de costo: %s\n'%costo)
#La funcion graficar datos recibe 1 matrix de Xs y 1 vector Y, ademas de una serie de thetas para generar una recta
graficarDatos(xDataCon1s,yData,thetas)

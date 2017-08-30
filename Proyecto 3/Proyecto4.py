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

#Hipothesis a usar, Vector vX con valores en X y vector thetas
def hipothesis(vX,thetas):
    return thetas.transpose().dot(vX)

#La funcion sigmoidal esta dada por 1/(1 + e^(-z))
#Donde z es el resultado de la hipothesis thetaT*x
def sigmoidal(z):
    return 1.0/(1.0+np.e**(-z))

#X y Y son vectores de misma size
#Thetas tiene size == cols de X
#Regresa el valor del costo y un arreglo de gradientes por cada theta
#Al finalizar el gradiente con datos prueba debemos obtener un 0.203
def funcionCosto(thetas, X, Y):
    if type(thetas).__module__ != np.__name__: thetas = np.array(thetas)
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    #Si size thetas > Xs se deberia agregar 1s a X
    if len(thetas) > len(X[0]): X = np.append(np.ones((X.shape[0],1)), X , axis=1) 
    #Inicializar outputs
    J, grad = 0, thetas.copy()
    m = len(X) 
    for i in range(0,m):
        sigEval = sigmoidal(hipothesis(X[i],thetas))
        pt = -1*Y[i] *np.log(sigEval)
        pf = (1-Y[i])*np.log(1-sigEval)
        J += pt - pf                
    J /= m                              #Multiplicar todo por 1/m, J ya esta
    
    #Calcular Gradiente del costo de thetas
    for j in range(0, len(thetas) ):
        grad[j] = sum( (sigmoidal(hipothesis(_x,thetas))-_y)*_x[j] for _x,_y in zip(X,Y) ) /m

    return (J, grad)

#Regresa un vector de thetas calculado usando gradiente desc
def aprende(theta,X,Y,iteraciones=1500):
    '''Validar que todo sea Numpy array/matrix'''
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(Y).__module__ != np.__name__: Y = np.array(Y)
    if theta is None: theta = np.zeros(X.shape[1]+1)  #Si no nos dieron theta llenarla con numero de Xs+1
    elif type(theta).__module__ != np.__name__: theta = np.array(theta)

    #Agregar ones a X
    fixedX = np.append(np.ones((X.shape[0],1)), X , axis=1) 
    m = len(X)
    for it in range(0,iteraciones): 
        tempThetas = theta.copy() #Thetas temporales
        for j in range(0, len(tempThetas) ):
            #FORMULA thetaI = thetaI-(1/m)*sum( (sigmoidal(hip(xi))-yi)*x[j]i  )
            s = sum( ( sigmoidal(hipothesis(_x,theta)) -_y )*_x[j] for (_x,_y) in zip(fixedX,Y) )
            tempThetas[j] = theta[j] - (s/m)
        theta = tempThetas
    return theta

def predice(theta,X):
    if type(X).__module__ != np.__name__: X = np.array(X)
    if type(theta).__module__ != np.__name__: theta = np.array(theta)
    
    #Si size thetas > Xs se deberia agregar 1s a X
    if len(theta) > len(X[0]): X = np.append(np.ones((X.shape[0],1)), X , axis=1) 
    
    p = np.zeros(len(X)) #Crear un vector de zeros para la respuesta
    f = lambda xi: sigmoidal( hipothesis(  xi,theta) ) #Esta es la prediccion, el resultado de la sigmoidal
    for i in range(0,len(X)): 
        p[i] = 1 if f(X[i]) >= 0.5 else 0
    return p

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

#Falta graficar la sigmoidal
def graficaDatos(X,Y,theta):
    for _x,_y in zip(X,Y): plt.scatter( _x[0],_x[1], marker="x" if _y else 'o' ) #Puntos X/Y

    #Nos dieron datos crudos pero nuestras thetas estan normalizadas
    #nX = normalizacionDeCaracteristicas(X)[0]
    x1_min = np.amin(X,axis=0)[0]
    x1_max = np.amax(X,axis=0)[0]
    #Dos valores es suficiente puesto que es un recta
    xs = [x1_min, x1_max]
    #0.5 es la brecha de cuando pasa o no pasa el examen
    f = lambda x1,th : (0.5-th[0]-th[1]*x1)/th[2]
    #Evaluar x2 por cada x1
    plt.plot( xs  , [f(xi,theta) for xi in xs] )
    #plt.legend() # Add a legend
    plt.show()   # Show the plot



#Ruben Cuadra A01019102
import matplotlib.pyplot as plt
import numpy as np
from random import uniform
import csv

def getDataFromFile(filename,delimiter=" "):  
    vals = []
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i,line in enumerate(reader):
            if not line[0]: line.pop(0) #Chars vacios
            vals.append( [float(v) for v in line] )
    return np.array(vals)

def graficarDatos(data,centroids_history):
    k = len(centroids_history[0]) #Numero centroides
    h = len(centroids_history)    #Cantidad de iteraciones

    for val in data:   #Graficar los datos en negro
        plt.scatter(*val, color="black")

    for i in range(k): #Graficar linea que une historial de centroides
        plt.plot( *np.array( [centroids_history[j][i] for j in range(h)] ).T ,color="blue" ) 

    for chistory in centroids_history:
        for centroid in chistory: #Poner X en centroides segun el historial
            plt.scatter(*centroid,marker="x",color="red")
    plt.show()

#Datos X
#Numero Clusters
#choice nos dice el algoritmo de seleccion, crear puntos o elegir de existentes
def kMeansInitCentroids(X, K, choice=False):
    centroids = []
    if choice:  #Elegir de los puntos existentes
        temp = X.copy()
        np.random.shuffle(temp)
        centroids = [temp[i] for i in range(K)]
    else: #Crear valores aleatorios
        maxs = np.amax(X,axis=0)
        mins = np.amin(X,axis=0)    
        for i in range(K): centroids.append( [ uniform(mins[0],maxs[0]),uniform(mins[1],maxs[1]) ] )
    return np.array(centroids)

#Datos X, arreglo de N+1 elementos donde cada elemento es un arreglo de [x,y]
#Centroides ya inicializados, es un arreglo de N elementos donde cada elemento es un arreglo de [x,y] , es decir numeros
def findClosestCentroids(X, i_centroids):
    idx = [ [] for i in range( len(i_centroids) ) ] #Init return 
    for ix,x in enumerate(X):
        ii = 0                  #Centroide mas cercano al punto x
        min_dist = float("inf") #Distancia entre el punto x y ese centroide
        for i,c in enumerate(i_centroids): #index, centroide
            dist = np.linalg.norm(x-c)     #Distancia de la x actual con ese centroide
            if dist<min_dist:              #Distancia a este punto es menor a la minima, actualizar
                ii       = i
                min_dist = dist 
        
        idx[ii].append(ix) #ya que calculamos a que centroide pertenece, agregar indice de esa X

    return np.array( idx )

#X son los datos iniciales
#idx es un arreglo de K elementos, cada elemento posee un arreglo de indices de X 
#K es la size de idx
def computeCentroids(X,idx,K):
    centroids = [ [0,0] for _ in range( K ) ] #Inicializar retorno
    
    for i,cluster in enumerate(idx):
        new_x,new_y,elements = 0,0,len(cluster) #Coordenadas para este nuevo cluster, elementos
        
        for p in cluster:
            new_x += X[p][0] 
            new_y += X[p][1]

        centroids[i][0] = new_x/elements
        centroids[i][1] = new_y/elements

    return np.array(centroids)

def runkMeans(X,centroids, max_iters, plot=False):
    k = len(centroids)
    if plot: history = [centroids] #Inicializar con 1 elemento
    for i in range(max_iters):
        idx = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,idx,k)
        if plot: history.append(centroids)
    if plot: graficarDatos(data, np.array(history))
    return centroids

if __name__ == '__main__':
    data = getDataFromFile("ex7data2.txt")
    k,iters = 3,10
    # centroids = kMeansInitCentroids(data,k,choice=False) # 3 clusters
    centroids = np.array([[3,3],[6,2],[8,5]]) # 3 clusters
    centroids = runkMeans(data,centroids,iters,plot=True)
    

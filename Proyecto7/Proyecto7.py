#Ruben Cuadra A01019102
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import math, csv, copy
from random import random, uniform
from enum import Enum
import json, sys

def getDataFromFile(filename,delimiter=" "):  
    vals = []
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for i,line in enumerate(reader):
            if not line[0]: line.pop(0) #Chars vacios
            vals.append( [float(v) for v in line] )
    return np.array(vals)

def graficarDatos(data,centroids):
    for val in data:
        plt.scatter(*val)
    
    for c in centroids:
        plt.scatter(*c,marker="x",color="red")
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

        new_x /= elements
        new_y /= elements

        centroids[i][0] = new_x
        centroids[i][1] = new_y

    return np.array(centroids)

def runkMeans(X,centroids, max_iters, plot=False):
    k = len(centroids)
    for i in range(max_iters):
        idx = findClosestCentroids(X,centroids)
        centroids = computeCentroids(X,idx,k)
    #graficarDatos(data,centroids)
    return centroids

if __name__ == '__main__':
    data = getDataFromFile("ex7data2.txt")
    k = 3
    # centroids = kMeansInitCentroids(data,k,choice=False) # 3 clusters
    centroids= np.array([[3,3],[6,2],[8, 5]]) # 3 clusters
    centroids = runkMeans(data,centroids,10,plot=False)
    print centroids
    

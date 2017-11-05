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

def findClosestCentroids(X, i_centroids):
    idx = [None]*len(i_centroids)


    for x in X:
        print i_centroids[0]
        print x
        
        dist = np.linalg.norm(x-i_centroids[0])
        print dist
        break


    return idx

if __name__ == '__main__':
    data = getDataFromFile("ex7data2.txt")
    centroids=kMeansInitCentroids(data,3,choice=False) # 3 clusters
    # initial_centroids=[[3,3],[6,2],[8, 5]] # 3 clusters
    for i in range(1):
        idx = findClosestCentroids(data,centroids);
        print idx

    #graficarDatos(data,centroids)
    



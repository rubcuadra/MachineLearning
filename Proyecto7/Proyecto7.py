#Ruben Cuadra A01019102
#import gnumpy as gpu
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import numpy as np
import math, csv, copy
from random import random
from enum import Enum
import json, sys

def getDataFromFile(filename,delimiter=" "):  
    val = [ [],[] ]
    with open(filename,'r') as f:
        reader = csv.reader(f, delimiter=delimiter)
        for c,i in enumerate(reader):
            if i: #Caracteres ''
                val[0].append( [float(j) for j in i[0: len(i)-1]])
                val[1].append( float( i[-1] )  )
    return [np.array(val[0]),np.array(val[1])]
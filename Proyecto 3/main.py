from Proyecto4 import *

if __name__ == "__main__":
    #Leer archivos
    xData, yData = getDataFromFile('data.csv')
    nX = normalizacionDeCaracteristicas(xData)[0]
    thts = aprende(None,nX,yData)
    
    print funcionCosto(thts,nX,yData)
    #print predice(thts,xData[3])
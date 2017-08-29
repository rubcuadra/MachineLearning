from Proyecto4 import *

if __name__ == "__main__":
    #Leer archivos
    xData, yData = getDataFromFile('data.csv')
    nX = normalizacionDeCaracteristicas(xData)[0]
    nY = normalizacionDeCaracteristicas(yData)[0]
    thts = aprende(None,xData,yData)
    #print thts
    #print predice(thts,xData[1])
from Proyecto3 import *

if __name__ == "__main__":
    xData, yData = getDataFromFile('datos.csv')
    nX, mediasX, sigma = normalizacionDeCaracteristicas(xData)
    
    jHistoriaGradiente, thetasGradiente  = gradienteDescendenteMultivariable(nX,yData,alpha=0.2) 
    thetasEcuacion = ecuacionNormal(xData,yData)
    
    print thetasGradiente
    print thetasEcuacion

    # print "{:,}".format( predicePrecio(xData[0],thetasNorm) ) 
    # print "{:,}".format( predicePrecio(xData[0],thetas2) ) 

    graficaError(jHistoriaGradiente)
from Proyecto3 import *

def buscarAlfa(xNormalizadas,Y,alpha=0.001,iteraciones=50, multiplicador=3):
    r, alph = float("inf"), alpha
    while alpha<2:
        jHistoriaGradienteNormalizado, thetasGradienteNormalizado  = gradienteDescendenteMultivariable(nX,yData, alpha=alpha ,iteraciones=50)
        if jHistoriaGradienteNormalizado[-1] < r:
            alph = alpha
            r = jHistoriaGradienteNormalizado[-1]
            #graficaError(jHistoriaGradienteNormalizado)    
        alpha *= multiplicador
    return alph, r #Regresa alpha y el ultimo costo 

if __name__ == "__main__":
    #Leer archivos
    xData, yData = getDataFromFile('datos.csv')
    thetasEcuacion = ecuacionNormal(xData,yData)
    print ('Thetas de la ecuacion normal ', thetasEcuacion)
    if len(xData[0]) > 1: #Se debe normalizar
        nX, mediasX, sigma = normalizacionDeCaracteristicas(xData)  #Normalizar X
        nY, mediasY, sigmaY = normalizacionDeCaracteristicas(yData) #Normalizar Y
        jHistoriaGradiente, thetasGradiente  = gradienteDescendenteMultivariable(nX,nY, alpha=0.003 ,iteraciones=1500)
        print ('Thetas usando gradiente descendente multivariable normalizado ', thetasGradiente)
        
        fixedX = np.append(np.ones((nX.shape[0],1)), nX , axis=1) #Agregar ones a X
        print (calculaCosto( fixedX , nY ,thetasGradiente))         #Comparar vs que?

        graficaError(jHistoriaGradiente)    
    else:    
        jHistoriaGradiente, thetasGradiente  = gradienteDescendenteMultivariable(xData,yData)
        print ('Thetas usando gradiente descendente', thetasGradiente)

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
    #Normalizar
    nX, mediasX, sigma = normalizacionDeCaracteristicas(xData)
    jHistoriaGradienteNormalizado, thetasGradienteNormalizado  = gradienteDescendenteMultivariable(nX,yData, alpha=0.729 ,iteraciones=1500)
    thetasEcuacion = ecuacionNormal(xData,yData)
    
    print ('Thetas usando gradiente normalizado ', thetasGradienteNormalizado)
    print ('Thetas de la ecuacion normal ', thetasEcuacion)
    print ("Buena alpha: ",buscarAlfa(nX,yData)[0]) #Nos da el 0.729 buscado
    graficaError(jHistoriaGradienteNormalizado)    
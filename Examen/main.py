from Examen import *

if __name__ == "__main__":
    xData, yData = getDataFromFile('datos.csv')
    errors, thetas = gradienteDescendenteMultivariable(xData,yData,alpha=0.09,iteraciones=100)
    graficaError(errors)
    graficarDatos(xData,yData,thetas)
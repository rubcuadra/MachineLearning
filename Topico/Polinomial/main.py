from polinomial import *

def setColsTo(newNumRows, npMatrix):
	totalRecords = len(npMatrix)
	if totalRecords == 0: return npMatrix
	if newNumRows==0: return np.array([[0] for i in range(0,totalRecords)]) 
	currentCols = len(npMatrix[0])

	if newNumRows==currentCols:
		return npMatrix
	if newNumRows>currentCols: #Appendear a la derecha elevando el elemento 0
		for i in range(0+2,newNumRows-currentCols+2): #+2 para empezar a elevar
			b = np.array( [[j**i] for j in npMatrix[:,0]] )	#Obtener la nueva columna
			npMatrix = np.concatenate((npMatrix, b), axis=1) #Agregarsela al npMatrix
		return npMatrix
	#Sacar las primeras newNumRows y dejarlas
	return npMatrix[:,:newNumRows]

gradoDeseado = 3
xData,yData = getDataFromFile("datos.csv")
newXMatrix = setColsTo( gradoDeseado, xData )
thetas = gradienteDescendenteMultivariable(newXMatrix,yData,alpha=0.001,iteraciones=1000)
graficar(xData,yData,thetas, newXMatrix)
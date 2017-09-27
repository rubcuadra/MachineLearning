import numpy as np
from random import random,randint
#Encontrar el maximo en la ecuacion x^2 donde x solo puede ser mayor 
def mainFunc(x):
	return x^2

#Recibe un sujeto de la poblacion y lo evalua
#Regresa una marca dependiendo que tan bueno o malo pueda ser para la solucion
def fitnessFunc(subject):
	sValue = decode(subject) #Decodificar
	return sValue**2 #Un numero grande

#Vector de Numpy array
#Con valores binarios -> Decimal
def decode(v,base=2): 
	num = "".join( str(i) for i in v ) #
	return int(num,base)

#Valor decimal
#Decimal -> np.Array con valores decimales
def encode(value,base=2,bits=0):
	num = np.base_repr(value,base)
	return np.array(num.zfill(bits)) #Fill with 0s and to npArray

#Los especimens solo existen en 0<=x<=31
def getRandomSpecimen(bits):
	val = randint(0,31)
	return val

#Recibe: 
#	Una poblacion (Arreglo de especimenes) 
#	Sus Probabilidades de sobrevivencia (Calculadas con fitnesFunct)
#Devuelve: 
#	Arreglo de 2 posiciones representando especimenes
def getParents(p,spr):
	parents = []
	while len(parents) < 2:               #Solo hay 2 padres
		acum = spr.cumsum()               #Probab acumulada
		choice = random()*acum[-1]	      #Obtener un numero al azar y ajustar
		for i,(specimen,rate) in enumerate(zip(p,acum)):
			if choice <= rate:	          #Comparar con acumulada,sacar specimen
				parents.append(specimen)  #Agregar como padre
				p = np.delete(p,i)        #Remover de specimens seleccionables
				spr = np.delete(spr,i)    #Sacar esa opcion
				break
	return parents

#Recibe:
#	Probabilidad de que se reproduzcan 2 especimenes
#	Especimen 1
#	Especimen 2
#Devuelve
#	Los hijos o Nada
def reproduce(probabCruce,mom,dad):
	if random() < probabCruce: #Se pueden reproducir
		toCut = randint(1,min(len(mom),len(dad))-1) #Ni al inicio ni al final
		son1 = mom[:toCut]+dad[toCut:]
		son2 = dad[:toCut]+mom[toCut:]
		return [son1,son2]
	else:
		return None

def mutate(char,probability):
	if random()<probability:
		return "1" #Podria ser un random en los simbolos restantes
	else:
		return char

#Recibe:
#	Probabilidad de mutar
#	Hijo 1
#	Hijo 2
#Devuelve 
#	Arreglo de 2 posiciones representando especimenes(Hijos)
def mutateSons(pm,son,daughter):
	son = "".join( mutate(l,pm) for l in son)
	daughter = "".join(mutate(l,pm) for l in daughter)
	return [son,daughter]

#Para guardar a lo largo de las generaciones
def specimenToDict(specimen,gen):
	return {"value":specimen,"score":fitnessFunc(specimen),"gen":gen}

#Algoritmo genetico, recibe: 
#	Cantidad de especimenes en la muestra
#	Bits de cada especimen
#	Probabilidad de reproducirse
#	Probabilidad de que un bit mute
#	Numero de generaciones
def genetic(popSize,specBits,probabCruce=1,probabMutacion=0.001,generaciones=10):
	#Some funcs
	ff = np.vectorize( fitnessFunc )         										 
	sp = np.vectorize( lambda s,totalPoints: np.divide(s,totalPoints,dtype=float)  ) 
	#initial Population
	population   = np.array( [ encode( getRandomSpecimen(specBits),bits=specBits ) for i in range(popSize)] )
	bestSpecimen = specimenToDict(population[0],0)
	#Pasar por generaciones
	for gen in range(generaciones):
		#Get Scores and Probabilities
		specimenSurvival = ff( population ) 
		genBestSpecimen  = specimenToDict(population[np.argmax(specimenSurvival)],gen=gen)
		specimenSurvivalRate = sp( specimenSurvival, specimenSurvival.sum() )	
		newPopulation = [] #Aqui se llenara
		while len(newPopulation) < popSize:
			parents = getParents(population,specimenSurvivalRate) 
			sons    = reproduce(probabCruce,*parents) #Intentar reproducir
			if sons == None: continue  #Obtener nuevos padres
			sons    = mutateSons(probabMutacion,*sons)	  #Validar que nadie muto
			newPopulation.extend(sons)    #Agregar hijos como nueva generacion
		
		#Eliminar si hay hijos de mas
		if len(newPopulation)==popSize+1: newPopulation.pop()
		newPopulation = np.array(newPopulation)
		population = newPopulation
	
		if genBestSpecimen["score"] > bestSpecimen["score"]:
			bestSpecimen = genBestSpecimen 

	return bestSpecimen

if __name__ == '__main__':
	specBits = 5
	numIndividuos = 8

	bestSpecimen = genetic(numIndividuos,specBits)
	print decode(bestSpecimen["value"])

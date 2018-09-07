from Proyecto2 import busquedaAstar

if __name__ == '__main__':
    edoInicial  = [
    	[0, 1, 2], 
    	[4, 5, 3], 
    	[7, 8, 6]
    ]
    
    edoFinal = [
    	[1, 2, 3], 
    	[4, 5, 6], 
    	[7, 8, 0]
    ] 

    steps = busquedaAstar(edoInicial, edoFinal, 1) # puede llamarse con 1
    print (steps)
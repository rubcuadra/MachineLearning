import numpy as np
from scipy.misc import imread
from pickle import load, dump
from random import shuffle
import matplotlib.pyplot as plt

def hopefield(patterns):
    s = np.prod(images[0].shape) # (28,28) = 784 ; (5,) = 5
    matrix = np.zeros( (s,s) )
    p = 0
    for pattern in patterns:
        v = stateToVector(pattern) #Image/Pattern to vector for learning
        for i in range(s):
            for j in range(i): #Iterate only lower diagonal:
                matrix[i,j] += (2*v[i]-1)*(2*v[j]-1) #Update
        print (f"{p}/{len(patterns)}")
        p+=1
    matrix = matrix + matrix.T #Convert Diagonal -> Full
    return matrix

def norm(matx, t): return np.where(matx >= t, 1, 0)

def stateToVector(state):
    if len(state.shape)==1: 
        return state
    else:
        temp = state.copy()
        temp.resize( np.prod( state.shape ) )
        return temp

def vectorToState(vector,shape):
    if len(shape)==1: 
        return vector
    else:
        return vector.copy().reshape(shape)

#Recive 1 initial state/picture/pattern and the Hopefield matrix(weights)
def getPattern(_state,hf,t, max_iters=1000):
    shape = _state.shape      #Remember original dimensions
    v = stateToVector(_state) #Images to vector
    order = [i for i in range(len(v))] #Init random order
    changed = 1 #do_while
    while changed and max_iters: #While something changed (changed>0)
        changed = len(order) #Reset decreaser
        shuffle(order)       #new update order
        for ix in order:     #Iterate over neurons for updates
            nV = norm( hf[ix]@v ,t) #New value with dot product, possible because diagonal of 0s; norm puts 1s or 0s
            if nV == v[ix]:
                changed-=1 #If it falls *changed* times it means that nothing changed and we finished    
                continue
            v[ix] = nV
        max_iters-=1
    return vectorToState(v,shape)  #Return vector as original shape     

def plotImages(*imgs,cmap='gray'):
    ti = len(imgs)
    fig,axs = plt.subplots(1,ti,sharex=True,sharey=True)
    axs = [axs] if ti==1 else axs #For better iteration
    for im,ax in zip(imgs,axs):
        ax.axis("off")
        ax.imshow(im,cmap='gray')
    plt.show()

def initWe(patterns):
    s = np.prod(images[0].shape) # (28,28) = 784 ; (5,) = 5
    matrix = np.zeros( (s,s) )
    
    pts = [stateToVector(p) for p in patterns] #Convert to vectors
    for i in range(s):
        for j in range(s):
            if i == j: continue #started in 0
            matrix[i,j] = sum( xk[i]*xk[j] for xk in pts )

    return matrix

if __name__ == '__main__':
    testImages = True
    
    if testImages: 
        images = [ imread("corr/0.png", mode="L"), imread("corr/1.png", mode="L")]
        images = [ np.where(i > 0, 1, 0) for i in images ]
        states = [ imread(f"corr/s{i+1}.png", mode="L") for i in range(6) ]
        states = [ np.where(i > 0, 1, 0) for i in states ]
    else:
        #NON-IMAGES
        images = np.array([ [0,1,1,0,1] , [1,0,1,0,1] ])
        states = [ np.array([1,1,1,1,1]) ]

    #Learn Patterns
    if True:
        hopefield_matrix = hopefield(images) 
        with open(f'temp.uu','wb+') as dp: dump(hopefield_matrix,dp) #Save
    else:
        with open(f'nums.uu','rb') as dp: hopefield_matrix = load(dp) #Load
    
    # plotImages(hopefield_matrix)
    # print(hopefield_matrix.min())
    # print(hopefield_matrix.max())
    # hopefield_matrix=initWe(images) #According to slides
    
    results = []
    for state in states:
        result = getPattern(state, hopefield_matrix, 0 )
        results.append(result)
    if testImages:
        plotImages(*states)
        plotImages(*results)
    else:
        print(results)

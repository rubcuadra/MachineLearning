from sklearn import datasets
from sklearn.preprocessing import normalize
from sklearn.datasets import fetch_mldata
import numpy as np
from random import randint
from keras.datasets import mnist
import matplotlib.pyplot as plt
import matplotlib.animation as animation


class Hopfield:
    def __init__(self):
        pass

    def fit(self, patterns):
        self.data = patterns
        self.dim = len(patterns[0])
        self.weights = np.zeros((self.dim, self.dim))
        self.n_patterns = len(self.data)
        mean = np.sum([np.sum(x) for x in self.data])/(self.n_patterns * self.dim)

        for i in range(self.n_patterns):
            t = self.data[i] - mean        #Normalize by mean
            self.weights += np.outer(t, t) #All vs All
        self.weights /= self.n_patterns    #W avg
        np.fill_diagonal(self.weights,0)   #Diagonal must be zero

    def predict(self, states):
        results = np.zeros(states.shape)

        for i,s in enumerate(states):
            results[i] = self.weights@s 

        return np.sign(results)


class Mnist:

    def __init__(self, digit=0, n_patterns=10):
        self.n = int(digit*7000)
        self.data = self.get_data(n_patterns)

    def get_data(self, patterns):
        mnist = fetch_mldata('MNIST original', data_home='mnist/')
        mnist.data = mnist.data.astype(np.float32)
        mnist.data /= 255
        return mnist.data[self.n:self.n+patterns]

    def add_noise(self, error_rate):
        data = self.data[:]
        for i, t in enumerate(data):
            s = np.random.binomial(1, error_rate, len(t))
            for j in range(len(t)):
                if not s[j]:
                    t[j] *= -1
        return data

def plotImages(*imgs):
    ti = len(imgs)
    fig,axs = plt.subplots(1,ti,sharex=True,sharey=True)
    axs = [axs] if ti==1 else axs #For better iteration
    for im,ax in zip(imgs,axs):
        ax.axis("off")
        ax.imshow(im)
    plt.show()

if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    norm = np.vectorize(lambda v:v/255)
    imgToVector = lambda img: np.resize(img, np.prod( img.shape ) )
    vecToImage  = lambda vec: np.reshape(vec, (28,28)) #Only for mnist, change dimensions for diff pics

    #Get images to train, get 70000p from the whole data [0<p<1]
    images = [] 
    classes = 2
    p = 0.0005
    num = 70000*p #Examples to take from the 70,000 - 100%
    perClass = int(num/classes)
    for tag in range(classes):
        for imgI in np.argwhere(y_train==tag)[:perClass]:
            images.append( norm(x_train[imgI])   ) #Normalize and add to images to test
    
    #Get Images to test
    args0 = np.argwhere( y_test==0 ).flatten()
    args1 = np.argwhere( y_test==1 ).flatten()
    states0  = [ norm( x_test[ args0[i] ] ) for i in range(3)] #NumOfElementsToTest
    states1  = [ norm( x_test[ args1[i] ] ) for i in range(2)] 
    states   = states0 + states1 
    
    #Convert Image to patterns
    train_data = np.array([ imgToVector(img) for img in images ])
    test_data  = np.array([ imgToVector(img) for img in states ])

    model = Hopfield()
    model.fit(train_data)
    results = model.predict(test_data)

    plotImages( *states )
    plotImages( *[ vecToImage(v) for v in results ] )
    
    

    
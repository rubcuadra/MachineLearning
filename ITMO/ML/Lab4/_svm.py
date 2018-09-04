from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from random import randint

class Kernels():
    GAUSSIAN = "gaussian"
    LINEAR   = 'linear'
    POLY = "poly"
    def linear_kernel(x1, x2):            return x1@x2
    def rbf_kernel(X,Y,gamma=None):       raise Exception("Not implemented")
    def polynomial_kernel(x, y, p=3):     return (1 + x@y) ** p
    def gaussian_kernel(x, y, sigma=5.0): return np.exp(-norm(x-y)**2 / (2 * (sigma ** 2)))
    def getPolynomialKernel(degree):      return lambda x,y: Kernels.polynomial_kernel(x,y,p=degree)

class SVC:
    def __init__(self, C=1.0, kernel=Kernels.GAUSSIAN, tol=0.001, sigma=0.1, degree=None,threshold=0):
        self.kernel = kernel
        self.tol = tol
        self.sigma = sigma
        self.C = C
        self.t = threshold
        self.degree=degree

    def computeKernelMatrix(self,X):
        if self.kernel == Kernels.LINEAR:
            K = Kernels.linear_kernel(X,X.T)
        elif self.kernel == Kernels.POLY:
            K = Kernels.polynomial_kernel(X,X.T,p=self.degree)
        elif self.kernel == Kernels.GAUSSIAN:
            m, n = X.shape
            K = np.zeros((m, m))
            for i in range(m):
                for j in range(i+1):
                    K[j, i] = Kernels.gaussian_kernel(X[i, :], X[j, :], sigma=self.sigma)
                    K[i, j] = K[j, i]
        else: raise Exception("Wrong kernel")
        return K

    def plot(self, points, labels): #Option to give test data or new data 
        config = {"c1.0":"black","m1.0":"x","c0.0":"red","m0.0":"o"}
        for point, label in zip(points, labels): plt.scatter(*point, c=config[f'c{label}'], marker=config[f'm{label}'])
        #Surface
        xx, yy = np.meshgrid(np.linspace(np.min(points[:, 0]), np.max(points[:, 0]), 100),np.linspace(np.min(points[:, 1]), np.max(points[:, 1]), 100))
        values = np.zeros(np.shape(xx))
        for i in range(np.size(xx, 1)): values[:, i] = self.predict(np.c_[xx[:,i],yy[:,i]]).ravel()
        #Plot
        plt.contour(xx, yy, values, colors='k')   
        plt.show()

    def fit(self, x, y, max_iters=20): #Iters without updates
        # self.sigma = 1/numFeatures
        X,Y = x.copy(),y.copy()
        m,n = X.shape
        Y = np.where(Y==0,-1,1) #Use 1 or -1
        Y = np.reshape(Y, (len(Y), 1))
        # Variables
        # What we look for are non-negative alpha coefficients
        self.alphas = np.zeros((m, 1))
        b = 0
        E = np.zeros((m, 1))
        passes = 0
        #Start
        K = self.computeKernelMatrix(X) #Kernels
        while passes < max_iters:
            num_changed_alphas = 0 #reseted on each iter
            for i in range(m):
                k = np.reshape(K[:, i][:], (len(K[:, i]), 1)) #(118,1) first col
                E[i] = b + np.sum(self.alphas*Y*k)-Y[i] #Ei = f(x(i) - y(i))
                if (Y[i]*E[i] < -self.tol) and\
                    (self.alphas[i]<self.C) or\
                    (Y[i]*E[i] > self.tol and self.alphas[i] > 0):
                    # Calculate 2 random weights (i and j) and until we get the best
                    j = randint(0, m-1)
                    while j == i: j = randint(0, m-1)
                    k = np.reshape(K[:, j][:], (len(K[:, j]), 1)) #(118,1) sec col
                    E[j] = b+np.sum(self.alphas*Y*k) - Y[j] #Ej = f(x(j) - y(j))
                    alpha_i_old = self.alphas[i].copy() #1 alpha
                    alpha_j_old = self.alphas[j].copy() #1 alpha
                    #Getting close, calc H,L
                    if Y[i] == Y[j]:
                        L = max(0, self.alphas[j] + self.alphas[i] - self.C)
                        H = min(self.C, self.alphas[j] + self.alphas[i])
                    else:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    eta = 2*K[i,j]-K[i,i]-K[j,j] #form
                    if H == L or eta >= 0: continue
                    #Update alpha 
                    self.alphas[j] =self.alphas[j]-(Y[j]*(E[i]-E[j]))/eta
                    #Choose critical points from updated
                    self.alphas[j] = max(L,  min(H, self.alphas[j]) )
                    #If delta alphas minor t, update and next
                    if abs(self.alphas[j] - alpha_j_old) < self.tol:
                        self.alphas[j] = alpha_j_old
                        continue
                    #Modify bias
                    self.alphas[i]=self.alphas[i]+Y[i]*Y[j]*(alpha_j_old-self.alphas[j])
                    b1 = b-E[i]-Y[j]*(self.alphas[i]-alpha_i_old)*K[i,j]-Y[j]*(self.alphas[j]-alpha_j_old)*K[i,j]
                    b2 = b-E[j]-Y[i]*(self.alphas[i]-alpha_i_old)*K[i,j]-Y[j]*(self.alphas[j]-alpha_j_old)*K[j,j]
                    if 0 < self.alphas[i] < self.C:   b = b1
                    elif 0 < self.alphas[j] < self.C: b = b2
                    else:                             b = (b1+b2)/2
                    num_changed_alphas += 1 #If the alpha changed reset iters
            if num_changed_alphas == 0: passes += 1
            else:                       passes = 0
        idx = (self.alphas > 0).ravel() #Relevants
        #Save results for predict
        self.X,self.tags,self.alphas,self.b = X[idx, :],Y[idx],self.alphas[idx],b
        self.w = (self.alphas*self.tags).T@self.X

    def predict(self, data):
        (m,n),(l,k) = data.shape, self.X.shape
        p = np.zeros((m, 1)) #Preditions
        if   self.kernel == Kernels.LINEAR: p = Kernels.linear_kernel(data,self.w.T) + self.b
        elif self.kernel == Kernels.POLY:   p = Kernels.polynomial_kernel(data,self.w.T,p=self.degree) 
        elif self.kernel == Kernels.GAUSSIAN:
            for i in range(m):
                p[i] = sum( Kernels.gaussian_kernel(data[i,:],self.X[j,:],sigma=self.sigma)*self.alphas[j]*self.tags[j] for j in range(l) ) + self.b
        f = np.vectorize(lambda v,t: 1 if v>=t else 0) #Threshold 0
        return f(p,self.t)

    def getConfusions(self, X, y):
        predictions = self.predict(X)
        TP,TN,FP,FN=0.,0.,0.,0.
        for prediction, target in zip(predictions, y):
            if prediction == 1:
                if target == 1:   TP += 1
                elif target == 0: FP += 1
            elif prediction == 0:
                if target == 1:   FN += 1
                elif target == 0: TN += 1
        #  T F
        #P
        #N  
        return np.matrix([[TP,FP],[FN,TN]])

if __name__ == '__main__':
    from sklearn.model_selection import train_test_split
    from sklearn.svm import SVC as skSVC
    import csv
    #Recibe como parametro el path a un archivo csv
    #con +2 columnas, "x,x1,x2,...xN,y" dond ela ultima es el valor en Y
    def getDataFromFile(filename,shuffle=False):  
        val = []
        with open(filename,'r') as f:
            reader = csv.reader(f, delimiter=',')
            for i in reader:
                val.append( [float(j) for j in i] )
        val = np.array(val)
        if shuffle: np.random.shuffle(val)
        return val[:,0:-1] , val[:,-1] #Y is the last column, else X

    X,tags = getDataFromFile("chips.txt")
    x_train, x_test, y_train, y_test = train_test_split(X,tags,test_size=0.1,random_state=0) 
    
    model = SVC(kernel=Kernels.GAUSSIAN, sigma=0.5) #POLY,LINEAR,GAUSSIAN. #degree=3
    model.fit(x_train, y_train)
    pred = model.predict(x_test) 
    print( f'Accuracy {accuracy_score(y_test, pred )}' )
    print(model.getConfusions(x_test,y_test))
    # model.plot(x_train,y_train)

    model = skSVC(C=1.0) #POLY,LINEAR,GAUSSIAN. #degree=3
    model.fit(x_train, y_train)
    pred = model.predict(x_test) 
    print( f'Accuracy {accuracy_score(y_test, pred )}' )
    # print(model.getConfusions(x_test,y_test))

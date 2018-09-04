from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score 
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.linear_model import LogisticRegression
from sklearn.manifold.t_sne import TSNE
from sklearn.svm import SVC
from pickle import load, dump
from random import randint
from matplotlib.mlab import PCA as PCA2
import matplotlib.pyplot as plt
import numpy as np
import gist

def plotImages(*imgs):
    ti = len(imgs)
    fig,axs = plt.subplots(1,ti,sharex=True,sharey=True)
    axs = [axs] if ti==1 else axs #For better iteration
    for im,ax in zip(imgs,axs):
        ax.axis("off")
        ax.imshow(im)
    plt.show()

if __name__ == '__main__':
    #Calculate Descriptors
    # with open("veeimgdump.uu", "rb") as dp: images = load( dp )
    # X = np.array( [ gist.extract(i) for i in images ] ) #Without reshape is (960,) .reshape(4,3,80)
    # with open('descriptors.uu','wb+') as dp: dump(X,dp)
    
    #Read files
    with open("descriptors.uu","rb")  as dp: data    = load( dp )
    with open("veetrgtdump.uu","rb")  as dp: tags    = load( dp )
    
    # best = [float("-inf"),0]
    for _ in range(1): #Multiple points for multiple local maximums
        seed = 2        #Controlled randomization #1817
        components = 160               #PCA DimensionalReduction 
        print(f"Seed {seed}")

        #Dimensional Reduction
        DimReductionMethods = (\
            PCA(n_components=components),\
            TSNE(n_components=2,n_iter=5000, random_state=seed),\
            # LDA(n_components=components),\
        )

        for T in DimReductionMethods:
            print(f"DR {T.__class__.__name__}")
            
            X = T.fit_transform(data) #Get Data with reduced dimensions
            x_train, x_test, y_train, y_test = train_test_split(X,tags, test_size=0.1,random_state=seed) #CrossVal

            MLMethods = [\
                DecisionTreeClassifier(random_state=seed),\
                KNeighborsClassifier(n_neighbors=5),\
                SVC(kernel="linear",random_state=seed),\
                LogisticRegression()
            ] #Select ML methods to test

            for m in MLMethods:
                m.fit(x_train,y_train)
                p = m.predict(x_test)
                print(f"\t{ format(accuracy_score(y_test,p)*100,'.2f') }% {m.__class__.__name__}")
                # if accuracy_score(y_test,p)>best[0]: best[0],best[1] = accuracy_score(y_test,p),seed
    # print(best)


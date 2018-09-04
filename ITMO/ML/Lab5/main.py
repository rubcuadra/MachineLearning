'''
You should implement the feature selection algorithm based on the utility metric (the Filter method). 
Implement several utility metrics and compare their performance at classification tasks.    
https://en.wikipedia.org/wiki/Feature_selection
https://machinelearningmastery.com/an-introduction-to-feature-selection/
'''

#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html#sklearn.feature_selection.chi2
#http://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.mutual_info_classif.html#sklearn.feature_selection.mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif, chi2, SelectKBest
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import chi2_contingency
from glob import glob
import pandas as pd
import numpy as np

def test_model(model,x_train,y_train,x_test,y_test):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('\t',x_train.shape )
    print('\tAccuracy: ', accuracy_score(y_test, pred))
    print('\tF-score: ', f1_score(y_test, pred, average='macro'))

#Folder with files/structure
#   *_train.data
#   *_train.labels
#   *_valid.data
#   *_valid.labels
def loadData(path):
    X,Y,x,y = [],[],[],[]
    with open( glob(f"{path}/*_train.data")[0]   ,"r" ) as td: X = [ [int(v) for v in line.split()] for line in td ]
    with open( glob(f"{path}/*_train.labels")[0] ,"r" ) as td: Y = [ [int(v) for v in line.split()] for line in td ]
    with open( glob(f"{path}/*_valid.data")[0]   ,"r" ) as td: x = [ [int(v) for v in line.split()] for line in td ]
    with open( glob(f"{path}/*_valid.labels")[0] ,"r" ) as td: y = [ [int(v) for v in line.split()] for line in td ]
    return (np.matrix(X),np.matrix(Y).A1,np.matrix(x),np.matrix(y).A1) 

class VarianceThresh():
    def __init__(self, threshold=0):
        self.th = threshold

    def fit(self,data):
        v = np.var(data,axis=0).A1                  #Get variances as vector
        self.ixs = np.argwhere( v <= self.th )[:,0] #Get indexes to eliminate

    def transform(self,data):
        newData = []
        ixs = list(self.ixs.copy()) + [-1] #to finish
        c = ixs.pop(0)
        for i,col in enumerate(data.T):
            if i == c: c = ixs.pop(0) #new index to remove
            else:      newData.append( col.A1 ) #add
        return np.matrix(newData).T

class ChiSquare: #Determine whether there is a significant difference between the expected frequencies and the observed frequencies in one or more categories.
    def __init__(self, alpha = 0.5):
        self.alpha = alpha 

    def fit(self,data,Y):
        self.ixs = []
        for i, X in enumerate(data.T):
            dfObserved = pd.crosstab(Y,X.A1) 
            chi2, p, degrfree, expected = chi2_contingency(dfObserved.values) 
            # self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
            if p<self.alpha: self.ixs.append(i) #SignLev

    def transform(self,data):
        newData = []
        ixs = self.ixs + [-1] #to finish
        c = ixs.pop(0)
        for i,col in enumerate(data.T):
            if i == c: c = ixs.pop(0) #new index to remove
            else:      newData.append( col.A1 ) #add
        return np.matrix(newData).T

if __name__ == '__main__':
    Xtrain,Ytrain,Xtest,Ytest = loadData("arcene")

    #VT
    VT = VarianceThresh(threshold=5000) #5000
    VT.fit(Xtrain)
    vtX_train = VT.transform(Xtrain) #Apply Selections
    vtX_test  = VT.transform(Xtest)  #Apply Selections
    
    #CHI2
    CHI = SelectKBest(score_func=chi2, k=550) #SelectKBest(score_func=chi2, k=550) #ChiSquare(alpha=0.05)
    CHI.fit(Xtrain,Ytrain)
    CHIXtrain = CHI.transform(Xtrain)
    CHIXtest  = CHI.transform(Xtest)
    
    #Different ML Techniques
    MLT = [LogisticRegression(),RandomForestClassifier(),DecisionTreeClassifier(),SVC(kernel='linear')]

    for model in MLT:
        print(model.__class__.__name__)
        print("\tFULL")
        test_model( model, Xtrain, Ytrain, Xtest, Ytest )
        print("\tVarianceThreshold")
        test_model( model, vtX_train, Ytrain, vtX_test, Ytest )
        print("\tCHI^2")
        test_model( model, CHIXtrain, Ytrain, CHIXtest, Ytest )
            


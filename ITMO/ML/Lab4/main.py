'''
- It is required to write SVM algorithm without using any libraries 
- Count the f-measure 
- Count the confusion matrix using the developed model 
- Investigate what Wilcoxon test is
- Use this test to compare kNN and SVM algorithms and calculate the p-value. 
- You should use old dataset chips.txt (from the first lab). 
'''
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats.distributions import norm
from scipy.stats import rankdata
from _svm import SVC, Kernels
from _knn import KNN,exp
import numpy as np
import csv
import warnings
warnings.filterwarnings('ignore') 
np.random.seed(0)

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

def crossVal(data,tags,intervals=5):    
    inters = int( len(data)/intervals )
    for i in range(intervals):
        _f,_t = inters*i,inters*(i+1)
        if i == 0:             #First iteration 
            trainX = data[_t:]
            trainY = tags[_t:]
        elif i == intervals-1: #Last Iteration
            trainX = data[:_f]
            trainY = tags[:_f]
        else:                  #In the middle cases
            trainX = np.concatenate((data[:_f] , data[_t:]), axis=0)
            trainY = np.concatenate((tags[:_f] , tags[_t:]), axis=0)
        yield i,(trainX,data[_f:_t],trainY,tags[_f:_t])

#https://en.wikipedia.org/wiki/Wilcoxon_signed-rank_test
#https://en.wikipedia.org/wiki/Rank_correlation
#http://www.statisticssolutions.com/how-to-conduct-the-wilcox-sign-test/
#https://support.minitab.com/en-us/minitab-express/1/help-and-how-to/basic-statistics/inference/how-to/one-sample/1-sample-wilcoxon/interpret-the-results/key-results/
#A Wilcoxon signed-rank test is a nonparametric test that can be used to determine whether two dependent samples were selected from populations having the same distribution.
def wilcoxonTest(data1, data2):
    # Data are paired and come from the same population.
    # Each pair is chosen randomly and independently[citation needed].
    # The data are measured on at least an interval scale when, as is usual, within-pair differences are calculated to perform the test (though it does suffice that within-pair comparisons are on an ordinal scale).
    #RANK CORRELATION
    x,y = np.array(data1), np.array(data2)
    d = x - y
    d = np.compress(np.not_equal(d, 0), d, axis=-1) #Ignore dif 0
    n = len(d)
    # Computing statistic
    r = rankdata(d)
    W = min(np.sum((d > 0) * r), np.sum((d < 0) * r))
    if n == 0: p_value = float('nan')
    else: #p-value
        ex = n*(n+1)*0.25
        var = n*(n+1)*(2* n + 1)/24
        z = (W-ex)/np.sqrt(var)
        p_value = 1 - 2. * norm.sf(abs(z)) #two-tailed significance
    return W, p_value 
    #the rank correlation r is equal to the test statistic W divided by the total rank sum S, or r = W/S. Using the above example, the test statistic is W = 9. The sample size of 9 has a total rank sum of S = (1 + 2 + 3 + 4 + 5 + 6 + 7 + 8 + 9) = 45. Hence, the rank correlation is 9/45, so r = 0.20.
    #Therefore we can reject our null hypothesis (with p < .002) that the average difference of the two observed measurements is 0.
    #P-value ≤ α: The difference between the medians is significantly different (Reject H0)
    #P-value > α: The difference between the medians is not significantly different (Fail to reject H0)

if __name__ == '__main__':
    X,tags = getDataFromFile("chips.txt",shuffle=True)
    # x_train, x_test, y_train, y_test = train_test_split(X,tags,test_size=0.1,random_state=0) 
    # 
    print("Loaded Data")
    svcf,svca,knnf,knna = [],[],[],[]
    for i,(x_train, x_test, y_train, y_test) in crossVal(X,tags,intervals=30):
        print(f"Cross Validation {i}")
        model = SVC(kernel=Kernels.GAUSSIAN) #POLY,LINEAR,GAUSSIAN. #degree=3
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        svcf.append( f1_score(y_test, pred, average='macro') )
        svca.append( accuracy_score(y_test, pred) )
        
        model = KNN(k=3, kernel=exp)
        model.fit(x_train, y_train)
        pred = model.predict(x_test)
        knnf.append( f1_score(y_test, pred, average='macro') )
        knna.append( accuracy_score(y_test, pred) )
        print( f"\tSVC\n\t\tf-score: {svcf[-1]}\n\t\taccuracy: {svca[-1]}" )
        print( f"\tKNN\n\t\tf-score: {knnf[-1]}\n\t\taccuracy: {knna[-1]}" )

    print (f"\n{'-'*15}RESULTS{'-'*15}")
    print( f"SVC\n\tf-score: {np.average(svcf)}\n\taccuracy: {np.average(svca)}" )
    print( f"KNN\n\tf-score: {np.average(knnf)}\n\taccuracy: {np.average(knna)}" )

    #Wilcoxon Test
    stat, p = wilcoxonTest(knna, svca) #Pass the historical accuracies
    print('W: ', stat, ' p: ', p)



import pandas
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.svm import SVC, LinearSVC


def combine_string(arr):
    return " ".join(arr)


def get_ingredients_freq(dataset):
    freq = {}

    for row in dataset:
        for ing in row:
            if ing in freq.keys():
                freq[ing] += 1
            else:
                freq[ing] = 1


    return freq

    # index = list(range(0, len(ingredients)))

    # frequencies = pandas.concat([pandas.Series(ingredients), pandas.Series(amount)], axis=1)
    # frequencies.columns = ['ingredient', 'amount']
    # frequencies.to_csv('ing_frequencies.csv', index=False)


def get_k_most_frequent_ingredients(dataset, k):
    frequencies = get_ingredients_freq(dataset)

    most_common = []

    for i in range(k):
        ing = max(frequencies, key=frequencies.get)
        most_common.append(ing)
        del frequencies[ing]

    return most_common


def test_solvers(c=1.0):
    solvers = ['newton-cg', 'sag', 'saga', 'liblinear', 'lbfgs']
    for solver in solvers:
        model = LogisticRegression(C=c, solver=solver)
        model.fit(x_train, y_train)
        # model.fit(X_train_dtm, y_train_b)
        # model = SVC(C=1.0) # Gives ~0.53 acc
        # model.fit(x_train, y_train)
        # model = RandomForestClassifier(n_estimators=30)
        # model.fit(x_train, y_train)

        pred = model.predict(x_test)

        print('Solver: ', solver)
        print('Accuracy: ', accuracy_score(y_test, pred))
        print('F-score: ', f1_score(y_test, pred, average='macro'))


def test_model(model):
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    print('Accuracy: ', accuracy_score(y_test, pred))
    print('F-score: ', f1_score(y_test, pred, average='macro'))


def make_submission(model):
    model.fit(X_train_dtm, y_train_b)
    pred = model.predict(X_test_dtm)
    submission = pandas.concat([idx, pandas.Series(pred)], axis=1)
    submission.columns = ['id', 'cuisine']
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':

    # Load data
    train = pandas.read_json('data/train.json')
    X_train = train.ingredients
    y_train_b = train.cuisine
    idx_tr = train.id

    test = pandas.read_json('data/test.json')
    X_test = test.ingredients
    idx = test.id

    # get_ingredients_freq(X_train)
    # print(get_k_most_frequent_ingredients(X_train, 10))

    # Pre-process data
    X_train = X_train.apply(combine_string)
    X_test = X_test.apply(combine_string)

    common_ingredients = get_k_most_frequent_ingredients(train.ingredients, 1)
    stop_words = ['®', '%'].extend(common_ingredients)
    # Vectorize words
    # vect = CountVectorizer(strip_accents='unicode', analyzer='word', stop_words=stop_words)
    vect = TfidfVectorizer(stop_words=stop_words, binary=False, analyzer='word', max_features=6000, max_df=.555)

    X_train_dtm = vect.fit_transform(X_train)
    X_test_dtm = vect.transform(X_test)

    x_train, x_test, y_train, y_test = train_test_split(X_train_dtm, y_train_b)
    
    # for i in [0.4, 0.8, 1.0, 1.2, 1.4]:
    # for i in [1.4, 1.8, 2.0]:
    #     print('Testing for: ', i)
    #     model = LogisticRegression(C=i, solver='newton-cg', n_jobs=3)
    #     test_model(model)
    
    # model = MultinomialNB(alpha=1.1)
    # model = RandomForestClassifier(n_estimators=20, n_jobs=-1)
    # model = LogisticRegression(C=1.7, solver='newton-cg', n_jobs=-1)
    # for c in [0.4, 0.8, 0.5, 0.6, 0.7]:
    #     print('C: ', c)
    #     model = LogisticRegression(C=c, solver='newton-cg', n_jobs=-1)
    #     test_model(model)
    # test_solvers(c=0.8)
    model = LinearSVC(C=0.4)
    test_model(model)
    # make_submission(model)
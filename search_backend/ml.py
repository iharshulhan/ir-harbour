import sklearn.externals.joblib as joblib
import time
from sklearn.externals.joblib import Memory
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from scipy import sparse
from sklearn import model_selection
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import shuffle


def execute():
    start = time.time()
    # mem = Memory("./mycache")
    #
    #
    # @mem.cache
    # def get_data():
    #     data = load_svmlight_files(("../data/Fold1/train.txt", "../data/Fold1/test.txt"))
    #     return data

    X, old_y = np.load('data5.npy')
    y = []
    for i in old_y:
        y.append(int(i))
    X = np.array([np.array(xi) for xi in X])
    y = np.array(y)
    print('Started')

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
    # print(X_train[0:1, 0], y_train[0])
    # Feature Scaling
    from sklearn.preprocessing import Normalizer

    sc = Normalizer()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)
    seed = 7
    # prepare models

    models = [('RandomForest', RandomForestClassifier(n_estimators=300, criterion='entropy', random_state=seed))]

    # evaluate each model in turn
    results = []
    names = []
    scoring = 'accuracy'
    for name, model in models:
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        cv_results = model_selection.cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        results.append(cv_results)
        names.append(name)
        msg = "Training %s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        model.fit(X_train, y_train)
        joblib.dump(model, name, compress=9)
        print(msg)
        y_pred = model.predict(X_test)
        print(metrics.classification_report(y_test, y_pred))
        cm = metrics.confusion_matrix(y_test, y_pred)
        print(cm)

    end = time.time()
    print('Time elapsed:', end - start)
    fig = plt.figure()
    fig.suptitle('Algorithm Comparison')
    ax = fig.add_subplot(111)
    plt.boxplot(results)
    ax.set_xticklabels(names)
    plt.show()

    ## The line / model
    # plt.scatter(y_test, predictions)
    # plt.xlabel(“True Values”)
    # plt.ylabel(“Predictions”)
    # y_pred = classifier.predict(X_test)
    #
    # # Making the Confusion Matrix
    # from sklearn.metrics import confusion_matrix
    # cm = confusion_matrix(y_test, y_pred)

execute()
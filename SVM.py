from math import gamma
from numpy.core.function_base import logspace
from sklearn.datasets import make_blobs
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.model_selection import GridSearchCV


df = pd.read_csv('new.csv',header=None)
data = np.array(df)
y = data[:,-1]
X = data[:,:-1]
print(X)

"""
cross - validation
"""
C = np.logspace(0.1, 20, 30)
gama = np.logspace(0.1,5,10)
tuned_parameters = [{'C': C}]
####linear-svm  one vs one 
SVM = svm.SVC(kernel= 'linear',gamma='scale', decision_function_shape='ovo')
####lib-linear  one vs all
lin_clf = svm.LinearSVC()

n_folds = 5
clf = GridSearchCV(estimator=SVM, param_grid=dict(C=C),n_jobs=-1)
# clf = GridSearchCV(estimator=lin_clf, param_grid=dict(C=C),n_jobs=-1)
clf.fit(X,y)  
print(clf.best_score_)
print(clf.best_estimator_.C )
#clf = GridSearchCV(clf, tuned_parameters, cv=n_folds, refit=False)


"""
两类svm的训练和测试
"""
#clf = svm.SVC(kernel='rbf', gamma=0.7, C=0.5)
skf = StratifiedShuffleSplit(n_splits=1, test_size=0.89, random_state=0)
#  = train_test_split(X, target, test_size=size, random_state=42) #随机取样
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

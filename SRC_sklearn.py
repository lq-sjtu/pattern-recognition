from operator import xor
from numpy.core.numeric import NaN
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.decomposition import DictionaryLearning
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.decomposition import SparseCoder



def normlization(Dict):
    for i in range(Dict.shape[0]):
        norm = np.linalg.norm(Dict[i,:])
        for j in range(Dict.shape[1]):
            Dict[i,j] = Dict[i,j]/norm

    return Dict


"""
train_test_split 函数

"""
df = pd.read_csv('new.csv',header=None)
data = np.array(df)
y = data[:,-1]
X = data[:,:-1]





skf = StratifiedShuffleSplit(n_splits=1, test_size=0.7, random_state=0)
#  = train_test_split(X, target, test_size=size, random_state=42) #随机取样
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
print(X_train.shape)
print(X_test.shape)
# # for i in range(38):
# #     a = list(y_test)
# #     print(a.count(i))


coder = SparseCoder(
    dictionary=X_train, transform_algorithm='omp',transform_n_nonzero_coefs = 30  
)
tmp = coder.transform(X_test)
# print(z)
print(np.count_nonzero(tmp))

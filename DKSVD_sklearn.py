import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model

def unpack(Dict):
    W = Dict[:,-38:]
    Dict_ = Dict[:,:-38]
    for i in range(Dict_.shape[1]):
       norm = np.linalg.norm(Dict[:,i]) 
       for j in range(Dict_.shape[0]):
           Dict_[j.i] = Dict_[j,i] /norm
       for j in range(W.shape[1]):
           W[j,i] = W[j,i]/norm
    return Dict_,W
 
def normlization(Dict):
    for i in range(Dict.shape[0]):
        norm = np.linalg.norm(Dict[i,:])
        for j in range(Dict.shape[1]):
            Dict[i,j] = Dict[i,j]/norm

    return Dict


def dict_update(y, d, x, n_components):
    """
    使用KSVD更新字典的过程
    """
    for i in range(n_components):
        index = np.nonzero(x[i, :])[0]
        if len(index) == 0:
            continue
        # 更新第i列
        d[:, i] = 0
        # 计算误差矩阵
        r = (y - np.dot(d, x))[:, index]
        # 利用svd的方法，来求解更新字典和稀疏系数矩阵
        u, s, v = np.linalg.svd(r, full_matrices=False)
        # 使用左奇异矩阵的第0列更新字典
        d[:, i] = u[:, 0]
        # 使用第0个奇异值和右奇异矩阵的第0行的乘积更新稀疏系数矩阵
        for j,k in enumerate(index):
            x[i, k] = s[0] * v[0, j]
    return d, x



def date_pre(X,y,r=100):
    H = np.zeros(shape=(0,38))
    for i in y:
        tmp = np.zeros(shape=(1,38))
        tmp[0,int(i)] = r
        H = np.vstack((H,tmp))
    X_train = np.hstack((X,H))
    return X_train

def unpack(Dict):
    W = Dict[:,-38:]
    Dict_ = Dict[:,:-38]
    for i in range(Dict_.shape[1]):
       norm = np.linalg.norm(Dict[:,i]) 
       for j in range(Dict_.shape[0]):
           Dict_[j.i] = Dict_[j,i] /norm
       for j in range(W.shape[1]):
           W[j,i] = W[j,i]/norm
    return Dict_,W

"""训练"""
df = pd.read_csv('new.csv',header=None)
data = np.array(df)
y = data[:,-1]
X = data[:,:-1]
print(X.shape)

X_train = date_pre(X,y,r=100)
X = normlization(X)
"""训练和测试集划分"""

skf = StratifiedShuffleSplit(n_splits=1, test_size=0.6875, random_state=0)
#  = train_test_split(X, target, test_size=size, random_state=42) #随机取样
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]









#####主程序


u, s, v = np.linalg.svd(X_train.T)
n_comp = 200
dictionary = u[:, :n_comp]

max_iter = 500
tolerance = 1e-4
max_iter = 10

for i in range(max_iter):
    # 稀疏编码
    x = linear_model.orthogonal_mp(dictionary, X_train.T)
    e = np.linalg.norm(X_train.T - np.dot(dictionary, x))
    if e < tolerance:
        break
    dict_update(X_train.T, dictionary, x, n_comp)
 
sparsecode = linear_model.orthogonal_mp(dictionary, X_test.T)

dic,w = unpack(dictionary)

train_restruct = dic.dot(sparsecode)


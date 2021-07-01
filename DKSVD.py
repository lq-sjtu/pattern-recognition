
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn import linear_model


def normlization(Dict):
    for i in range(Dict.shape[0]):
        norm = np.linalg.norm(Dict[i,:])
        for j in range(Dict.shape[1]):
            Dict[i,j] = Dict[i,j]/norm

    return Dict


"""
DKSVD
"""

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

def omp(X_train,y,n_comp,k =50):
    r = y#残差
    t = 1 #迭代次数
    flag  =True
    flag_2 = True
    tmp_l = []
    A = np.empty(shape=(0,3))
    X_ = np.zeros(shape=(n_comp,1))
    index = 0 #相似度最大的index
    X_train_copy = X_train.copy()
    for _ in range(k):
        index = 0
        max = np.dot(X_train[0,:],r)
        for i in range(X_train.shape[0]):
            score = np.dot(X_train[i,:],r)
            if score > max:
                max = score 
                index = i
        for j in range(X_train_copy.shape[0]):
            if np.linalg.norm(X_train_copy[j,:]-X_train[index,:])< 1e-6:
                tmp_l.append(j)
        """
        计算更新残差
        """
        if flag:
            flag = False
            A = X_train[index,:]
        else:
            A= np.vstack((A,X_train[index,:]))
        if flag_2:
            X = np.dot(np.dot(1/np.dot(A,A.T),A),y.T)
            flag_2 = False
        else:
            X = np.dot(np.dot(np.linalg.inv(np.dot(A,A.T)),A),y.T)#最小二乘
        r = y - np.dot(A.T,X)
        X_train = np.delete(X_train,index,axis=0)
    i = 0
    for k in tmp_l:
        X_[k] = X[i+1]
    return X,r

def date_pre(X_train,y_train,r=0.05):
    H = np.zeros(shape=(0,38))
    for i in y_train:
        tmp = np.zeros(shape=(1,38))
        tmp[0,int(i)] = r
        H = np.vstack((H,tmp))
    X_train = np.hstack((X_train,H))
    return X_train

"""
获得训练完毕的W矩阵和字典
"""
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



def evaluate(W,Dict_,X_test,y_test):
    for i in range(X_test.shape[0]):
        X,r = omp(Dict_,X_test[i,:])
    label = np.dot(W,X)



    pass


"""训练"""
df = pd.read_csv('new.csv',header=None)
data = np.array(df)
y = data[:,-1]
X = data[:,:-1]
print(X.shape)
X = normlization(X)

"""训练和测试集划分"""

skf = StratifiedShuffleSplit(n_splits=1, test_size=0.6875, random_state=0)
#  = train_test_split(X, target, test_size=size, random_state=42) #随机取样
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]



#####主程序

X_train = date_pre(X_train,y_train,r=0.05)

u, s, v = np.linalg.svd(X_train.T)
n_comp = 200
dictionary = u[:, :n_comp]

max_iter = 500
tolerance = 1e-4

flag = True
for i in range(max_iter):
    # 稀疏编码
    for i in range(X_train.shape[0]):
        tmp,_ = omp(dictionary, X_train.T,n_comp=n_comp,k=50)
        if flag:
            x = tmp
            flag = False
        else:
            x = np.vstack((x,tmp))

    e = np.linalg.norm(X_train - np.dot(dictionary, x))
    if e < tolerance:
        d,x = dict_update(X_train, dictionary, x, n_comp)
        break
    elif i == max_iter:
        d,x = dict_update(X_train, dictionary, x, n_comp)
    dict_update(X_train, dictionary, x, n_comp)


for i in range(X_train.shape[0]):
        tmp,_ = omp(dictionary, X_test[i,:],n_comp=n_comp,k=50)
        if flag:
            x = tmp
        else:
            x = np.append((x,tmp))

dict,w = unpack(d)
np.dot(w,x)




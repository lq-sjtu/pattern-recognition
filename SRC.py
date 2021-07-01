from operator import xor
from numpy.lib.recfunctions import recursive_fill_fields
from sklearn.decomposition import DictionaryLearning
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

df = pd.read_csv('new.csv',header=None)
data = np.array(df)
y = data[:,-1]
X = data[:,:-1]
print(X.shape)


def normlization(Dict):
    for i in range(Dict.shape[0]):
        norm = np.linalg.norm(Dict[i,:])
        for j in range(Dict.shape[1]):
            Dict[i,j] = Dict[i,j]/norm

    return Dict


def omp(X_train,y_train,k,y):
    r = y#残差
    t = 1 #迭代次数
    flag  =True
    flag_2 = True
    A = np.empty(shape=(0,3))
    label = []#存放类别
    index = 0 #相似度最大的index
    for _ in range(k):
        index = 0
        max = np.dot(X_train[0,:],r)
        for i in range(X_train.shape[0]):
            score = np.dot(X_train[i,:],r)
            if score > max:
                max = score 
                index = i
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
        label.append(y_train[index])
        X_train = np.delete(X_train,index,axis=0)
        y_train = np.delete(y_train,index)
    return A,X,label,r

#评估准则为残差最小的类别
def classfication(X_test,y_test,k = 30):
    
    result = []
    for i in range(X_test.shape[0]):
        A,X,a,r = omp(X_train,y_train,k,X_test[i,:])
        s = set(a)
        res = 100
        for j in s:
            rr = np.linalg.norm((X_test[i,:] - np.dot(A[a==j,:].T,X[a==j])))
            if rr < res:
                cls = j
                res = rr
        result.append(cls)
    return result

def evaluate(results,y_test):
    count = 0
    for i in range(len(y_test)):
        if results[i] == y_test[i]:
            count += 1
    accuracy = count/len(y_test)
    return accuracy

X = normlization(X)
skf = StratifiedShuffleSplit(n_splits=1, test_size=0.8, random_state=0)
#  = train_test_split(X, target, test_size=size, random_state=42) #随机取样
for train_index, test_index in skf.split(X, y):
    #print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

print(X)
results = classfication(X_test,y_test,k = 30)
print(evaluate(results,y_test))




    



#C=np.dot(A,np.transpose(B))
#C=np.dot(np.transpose(B),np.transpose(A)).
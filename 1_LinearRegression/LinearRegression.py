# coding:utf-8
# @Time: 2022/7/7 9:42 下午
# @File: LinearRegression.py
# @Software: PyCharm
import numpy as np
from matplotlib import pyplot as plt

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def featureNormalization(X):
    X_norm = np.array(X)
    mu = np.mean(X_norm, 0) # 每列的平均值
    sigma = np.std(X_norm, 0) # 每列的方差
    for i in range(X.shape[1]):
        X_norm[:, i] = (X_norm[:, i]-mu[i])/sigma[i]
    return X_norm, mu, sigma


def plot_X1_X2(X):
    plt.scatter(X[:,0], X[:,1])
    plt.show()


def linearRegression(alpha=0.01, num_iters=400):
    data = loadtxtAndcsv_data("data.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]
    m = len(y)
    col = data.shape[1]

    X, mu, sigma = featureNormalization(X)
    plot_X1_X2(X)
    
    
if __name__ == '__main__':
    linearRegression()
    
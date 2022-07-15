# coding:utf-8
# @Time: 2022/7/7 9:42 下午
# @File: LinearRegression.py
# @Software: PyCharm

#  加载txt和csv文件
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.font_manager import FontProperties

def loadtxtAndcsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def featureNormalize(X):
    """
        均值方差归一化Feature
    :param X:
    :return: X_norm: 归一化后的X
        mu: 均值
        sigma：方差
    """
    X_norm = np.array(X) # X转换成numpy数组对象才可以进行矩阵运算

    mu = np.mean(X_norm, 0)
    sigma = np.std(X_norm, 0)
    for i in range(X.shape[1]):
        X_norm[:, i] = (X_norm[:,i]-mu[i])/sigma[i] # 归一化# 均值方差归一化（standardization）

    return X_norm, mu, sigma


def plot_X1_X2(X):
    """   画二维图    """
    plt.scatter(X[:,0], X[:,1])
    plt.show()


def loss(X,y, theta):
    m = len(y)
    J=0
    J = (np.transpose(X*theta-y))*(X*theta-y)/2*m  # MSE
    return J


def gradientDescent(X, y, theta, alpha, num_iters):
    """ 梯度下降算法 """
    m = len(y)
    n = len(theta)

    temp = np.matrix(np.zeros((n, num_iters))) # 存迭代计算的theta
    J_history = np.zeros((num_iters, 1))

    for i in range(num_iters):
        h = np.dot(X, theta)  # 预测值
        # 梯度更新计算
        temp[:, i] = theta - ((alpha/m)*(np.dot(np.transpose(X), h-y))) # 梯度计算
        theta = temp[:,i]
        J_history[i] = loss(X,y, theta)
    return theta, J_history


def plotJ(J_history, num_iters):
    x = np.arange(1, num_iters+1)
    plt.plot(x, J_history)

    # font = FontProperties(fname="/Library/Fonts/Songti.ttc", size=14)  # 解决windows环境下画图汉字乱码问题
    plt.xlabel(u"迭代次数")
    plt.xlabel(u"损失函数")
    plt.title(u"loss随迭代次数变化")
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.show()

def linearRegression(alpha=0.01, num_iters=400):
    print('加载数据')
    data = loadtxtAndcsv_data("data.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]
    m = len(y)
    col = data.shape[1] # 有多少列

    X, mu, sigma = featureNormalize(X)
    plot_X1_X2(X) # 画图

    X = np.hstack((np.ones((m,1)), X))  # X前加一列1

    print("执行梯度下降算法...")

    theta = np.zeros((col, 1)) # 要求的就是theta使得loss最小
    y = y.reshape(-1, 1) # 保证y是列
    theta, J_history = gradientDescent(X, y, theta, alpha, num_iters)

    plotJ(J_history, num_iters)
    return mu, sigma, theta


def predict(mu, sigma, theta):
    p = np.array([1650, 3])
    norm_predict = (p - mu)/sigma
    final_p = np.hstack(((np.ones(1)), norm_predict))
    result = np.dot(final_p, theta)
    print(result)
    return result


def testLinearRegresion():
    mu, sigma, theta = linearRegression()
    print(f"mu={mu}, sigma={sigma}, theta={theta}")
    predict(mu, sigma, theta)

if __name__ == '__main__':
    testLinearRegresion()
    
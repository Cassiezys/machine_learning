
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize



def sigmoid(z):
    h = np.zeros((len(z), 1))
    h = 1.0/(1.0+np.exp(-z))
    return h

def loadtxtAndCsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def plot_data(X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)

    plt.figure(figsize=(15,12))
    plt.plot(X[pos, 0], X[pos, 1], 'ro')
    plt.plot(X[neg, 0], X[neg, 1], 'bo')
    plt.title(u'两个类别散点图')
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    plt.show()


def mapFeature(x1, x2):
    """ 创作多些feature防止偏差大
    degree=2, project into 1,x1,x2,x1^2,x1x2,x2^2"""
    degree = 3
    out = np.ones((x1.shape[0], 1))  # 映射后的结果数组（取代X）
    for i in np.arange(1, degree+1):
        for j in range(i+1):
            tmp = x1**(i-j)*(x2**j)
            out = np.hstack((out, tmp.reshape(-1,1)))
    return out


def costFunction(initial_theta, X, y, initial_lambda):
    """ 加上了L2正则化 lambda/2m*sum{1<=j<n}theta^2_j"""
    m = len(y)
    J=0
    h = sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0

    tmp = np.dot(np.transpose(theta1), theta1)
    # J = (-np.dot(np.transpose(y),np.log(h))-np.dot(np.transpose(1-y),np.log(1-h))+tmp*initial_lambda/2)/m   # 正则化的代价方程
    J1 = -np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1-y), np.log(1-h))
    L2_norm = tmp*initial_lambda/2
    J = (J1+L2_norm)/m
    # J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1-y), np.log(1-h)) + tmp*initial_lambda/2)/m
    return J


def predict(X, theta):
    m = X.shape[0]
    p = sigmoid(np.dot(X, theta))

    for i in range(m):
        if p[i] > 0.5:
            p[i] = 1
        else:
            p[i] = 0
    return p


def plotDecisionBoundary(theta, X, y):
    pos = np.where(y==1)
    neg = np.where(y==0)
    plt.figure(figsize=(15,12))
    plt.plot(X[pos,0], X[pos,1], 'ro')
    plt.plot(X[neg,0], X[neg,1], 'bo')
    plt.title(u"决策边界")
    u = np.linspace(-1, 1.5, 50)
    v = np.linspace(-1, 1.5, 50)

    z = np.zeros((len(u), len(v)))
    for i in range(len(u)):
        for j in range(len(v)):
            z[i,j] = np.dot(mapFeature(u[i].reshape(1,-1), v[j].reshape(1,-1)),theta)
    z = np.transpose(z)
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']

    plt.contour(u,v,z,[0,0.01], linewidths=2.0)
    plt.show()



def logisticRegression():
    data = loadtxtAndCsv_data("data2.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]
    plot_data(X, y)
    X = mapFeature(X[:,0], X[:,1])
    initial_theta = np.zeros((X.shape[1], 1))
    initial_lambda = 0.1  # 正则化系数 0.01,0.1,1...
    J = costFunction(initial_theta, X,y, initial_lambda)
    print(J)
    # result_theta = optimize.fmin(costFunction, initial_theta, args=(X,y,initial_lambda))
    result_theta = optimize.fmin_bfgs(costFunction,initial_theta,fprime=gradient, args=(X,y,initial_lambda))
    """ 拟牛顿法Broyden-Fletcher-Goldfarb-Shanno:
        fprime指定costFunction的梯度
    """
    p = predict(X,result_theta)
    print(u'训练集上准确率为%f%%'%np.mean(np.float64(p==y)*100))

    X = data[:, 0:-1]
    y = data[:,-1]
    plotDecisionBoundary(result_theta, X,y)

def gradient(initial_theta, X,y, initial_lambda):
    m = len(y)
    grad = np.zeros((initial_theta.shape[0]))

    h = sigmoid(np.dot(X, initial_theta))
    theta1 = initial_theta.copy()
    theta1[0] = 0

    grad = np.dot(np.transpose(X), h-y)/m + initial_lambda/m*theta1
    return grad

if __name__ == '__main__':
    logisticRegression()
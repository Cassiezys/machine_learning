# coding:utf-8
# @Time: 2022/7/16 9:21 上午
# @File: oneVsAll_LogisticRegression_scikit-learn.py
# @Software: PyCharm

import numpy as np
import scipy.io as spio
from sklearn.linear_model import LogisticRegression


def logisticRegression_oneVsAll():
    data = loadMatData("data_digits.mat")
    X = data['X']
    y = data['y']
    y = np.ravel(y)  #

    model = LogisticRegression()
    model.fit(X, y)

    predict = model.predict(X)
    print("准确率：%f%%"%np.mean(np.float64(predict==y)**100))


def loadMatData(fileName):
    return spio.loadmat(fileName)

if __name__ == '__main__':
    logisticRegression_oneVsAll()
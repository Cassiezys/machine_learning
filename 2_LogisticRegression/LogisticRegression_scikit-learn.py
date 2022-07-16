# coding:utf-8
# @Time: 2022/7/16 9:01 上午
# @File: LogisticRegression_scikit-learn.py
# @Software: PyCharm
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def loadtxtAndCsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def logisticRegression():
    data = loadtxtAndCsv_data("data1.txt", ",", np.float64)
    X = data[:, 0:-1]
    y = data[:, -1]

    x_tain, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_tain)
    x_test = scaler.fit_transform(x_test)

    model = LogisticRegression()
    model.fit(x_train, y_train)

    predict = model.predict(x_test)
    right = sum(predict==y_test)
    # print(predict.shape)
    # predict = np.hstack((predict.reshape(-1,1), y_test.reshape(-1,1)))
    # print(predict)
    print('准确率为%f%%'%(right*100.0/predict.shape[0]))

if __name__ == '__main__':
    logisticRegression()
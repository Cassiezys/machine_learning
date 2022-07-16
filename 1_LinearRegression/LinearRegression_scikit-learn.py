import numpy as np
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler


def loadtxtAndCsv_data(fileName, split, dataType):
    return np.loadtxt(fileName, delimiter=split, dtype=dataType)


def linearRegression():
    data = loadtxtAndCsv_data("data.txt", ",", np.float64)
    X = np.array(data[:, 0:-1], dtype=np.float64)
    y = np.array(data[:, -1], dtype=np.float64)
    tx = np.array([1650, 3]).reshape(1, -1)
    scaler = StandardScaler()
    scaler.fit(X)
    print(tx.shape)
    x_train = scaler.transform(X)
    x_test = scaler.transform(tx)
    model = linear_model.LinearRegression()
    model.fit(x_train, y)

    result = model.predict(x_test)
    print(model.coef_)
    print(model.intercept_)
    print(result)

if __name__ == '__main__':
    linearRegression()

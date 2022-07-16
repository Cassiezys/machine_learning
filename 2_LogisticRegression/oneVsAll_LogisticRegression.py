import matplotlib.pyplot as plt
import numpy as np
import scipy.io as spio
from scipy import optimize



def loadMatData(fileName):
    return spio.loadmat(fileName)


def display_data(imgData):
    sum = 0
    ''' 显示100个数 
    - 初始化一个二维数组， 将每行的数据调整成图像的矩阵，放进二维数组，之后显示
    '''
    pad = 1
    display_array = -np.ones((pad + 10 * (20 + pad), pad + 10 * (20 + pad)))
    for i in range(10):
        for j in range(10):
            display_array[pad + i * (20 + pad):pad + i * (20 + pad) + 20,
            pad + j * (20 + pad):pad + j * (20 + pad) + 20] = (imgData[sum, :].reshape(20, 20, order="F"))
            sum += 1

    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.imshow(display_array, cmap='gray')
    plt.axis('off')
    plt.show()


def sigmoid(z):
    h = np.zeros((len(z), 1))
    h = 1 / (1.0 + np.exp(-z))
    return h


def costFunction(theta, X, y, Lambda):
    m = X.shape[0]
    h = sigmoid(np.dot(X, theta))

    theta1 = theta.copy()
    theta1[0] = 0  # 正则化从1开始,所以先复制一份theta再赋值为1
    tmp = np.dot(np.transpose(theta1), theta1)
    J = (-np.dot(np.transpose(y), np.log(h)) - np.dot(np.transpose(1 - y), np.log(1 - h)) + tmp * Lambda / 2) / m
    return J


def gradient(theta, X, y, Lambda):
    h = sigmoid(np.dot(X, theta))
    theta1 = theta.copy()
    theta1[0] = 0  # 正则化从1开始，之前也没有梯度
    m = len(y)
    grad = (np.dot(np.transpose(X), h - y) + Lambda * theta1) / m

    return grad


def oneVsAll(X, y, num_labels, Lambda):
    ''' 求每个分类的theta， 返回所有的all_theta'''
    m, n = X.shape
    all_theta = np.zeros((n + 1, num_labels))  # 每一列是对应的theta
    X = np.hstack((np.ones((m, 1)), X))  # 补上bias
    class_y = np.zeros((m, num_labels))  # 数据y对应的数字0-9，都映射为0/1关系
    initial_theta = np.zeros((n + 1, 1))

    for i in range(num_labels):
        tmp = np.int32(y == i)
        class_y[:, i] = tmp.reshape(1, -1)

    for i in range(num_labels):
        result = optimize.fmin_bfgs(costFunction, initial_theta, fprime=gradient, args=(X, class_y[:, i], Lambda))
        all_theta[:, i] = result.reshape(1, -1)

    all_theta = np.transpose(all_theta)
    return all_theta


def predict_ont_asALl(all_theta, X):
    m = X.shape[0]
    X = np.hstack((np.ones((m, 1)), X))
    h = sigmoid(np.dot(X, np.transpose(all_theta)))
    p = np.array(np.where(h[0, :] == np.max(h, axis=1)[0]))
    for i in np.arange(1, m):
        t = np.array(np.where(h[i, :] == np.max(h, axis=1)[i]))
        p = np.vstack((p, t))
    return p


def logisticRegression_oneVsAll():
    data = loadMatData("data_digits.mat")
    X = data['X']
    y = data['y']
    m, n = X.shape
    num_labels = 10

    rand_indices = [t for t in [np.random.randint(x - x, m) for x in range(100)]]  # 生成100个0-m的随机数
    display_data(X[rand_indices, :])

    Lambda = 0.1
    all_theta = oneVsAll(X, y, num_labels, Lambda)

    p = predict_ont_asALl(all_theta, X)
    print(u"准确率：%f%%" % np.mean(np.float64(p == y.reshape(-1, 1)) * 100))


if __name__ == '__main__':
    logisticRegression_oneVsAll()
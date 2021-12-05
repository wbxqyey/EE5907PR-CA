import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

# load data
dataset = sio.loadmat('/Users/zhouying/PycharmProjects/Q1/spamData.mat')
xtrain = dataset['Xtrain']
ytrain = dataset['ytrain']
xtest = dataset['Xtest']
ytest = dataset['ytest']

# data processing

# use log-transform
log_xtrain = np.log(xtrain + 0.1)
log_xtest = np.log(xtest + 0.1)

# Euclidean distance for each data in training data and testing data
def Euc_distance_fcn(x_train, x_test):

    distance_matrices = np.zeros(shape=(len(x_test), len(x_train)))
    for i in range(len(x_test)):
        for j in range(len(x_train)):
            distance_matrices[i][j] = (np.sum((x_test[i, :] - x_train[j, :]) ** 2)) ** 0.5   # Euclidean distance
    return distance_matrices

train_matrices = Euc_distance_fcn(log_xtrain, log_xtrain)
test_matrices = Euc_distance_fcn(log_xtrain, log_xtest)

# print(train_matrices)

train_Dist = np.argsort(train_matrices)
test_Dist = np.argsort(test_matrices)

# train_Dist

def error_rateFcn(train_dist, y_train, y_test, k):
    err = 0
    for i in range(len(train_dist)):
        sort_idx = train_dist[i, :k]

        prob_1 = np.sum(y_train[sort_idx] == 1)
        prob_0 = np.sum(y_train[sort_idx] == 0)

        if (prob_1 > prob_0):
            ypred = 1
        else:
            ypred = 0

        if ypred != y_test[i]:
            err += 1

    return (err / len(y_test)) * 100


K1 = np.arange(1, 11, 1)
K2 = np.arange(15, 105, 5)
K_array = np.concatenate((K1, K2), axis=0)

errRate_train = []
errRate_test = []
for i in range(len(K_array)):
    errRate_train.append(error_rateFcn(train_Dist,ytrain,ytrain,K_array[i]))
    errRate_test.append(error_rateFcn(test_Dist,ytrain,ytest,K_array[i]))


plt.plot(K_array, errRate_train, label='Training Data')
plt.plot(K_array, errRate_test, label='Testing Data')
plt.xlabel('K Values')
plt.ylabel('Error Rate %')
plt.title('Training and Testing Data Error Rates')
plt.legend()
plt.show()

for i in [0, 9, 27]:
    print('K =', K_array[i])
    print('training error:', str(errRate_train[i]) + '%')
    print('testing error:', str(errRate_test[i]) + '%')
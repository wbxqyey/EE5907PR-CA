import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.io as sio

# load data
data = sio.loadmat('/Users/zhouying/PycharmProjects/Q1/spamData.mat')
xtrain = pd.DataFrame(data['Xtrain'])
ytrain = pd.DataFrame(data['ytrain'])
xtest = pd.DataFrame(data['Xtest'])
ytest = pd.DataFrame(data['ytest'])

# data processing

# use log-transform
log_xtrain = np.log(xtrain + 0.1)
log_xtest = np.log(xtest + 0.1)

# binarization
bin_xtrain = xtrain.applymap(lambda x: 1 if x > 0 else 0)
bin_xtest = xtest.applymap(lambda x: 1 if x > 0 else 0)

# Lambda array
lambda1 = np.arange(1, 11, 1)
lambda2 = np.arange(15, 105, 5)
lambda_array = np.concatenate((lambda1, lambda2), axis=0)

# Include bias term
x_train_bias = np.concatenate((np.ones((len(log_xtrain), 1)), log_xtrain), axis = 1)
x_test_bias = np.concatenate((np.ones((len(log_xtest), 1)), log_xtest), axis = 1)

D = x_train_bias.shape[1]
N = x_train_bias.shape[0]

# Newton's method
def newton_method(x_train_bias, x_test_bias, lam):
    check = 1
    w = np.zeros((D, 1))
# Criteria for stop converging
    while check > 0.01:
        mu = 1 / (1 + np.exp(-(np.dot(x_train_bias, w))))
        G = x_train_bias.transpose().dot(mu - ytrain)
        S = np.diag(np.squeeze((mu * (1 - mu)), axis=1))
        H = np.dot(np.dot(x_train_bias.transpose(), S), x_train_bias)
        I = np.identity(D)
# The first row and column set to 0 for the identiry matrix
        I[0,0] = 0
        H_reg = H + lam * I
# Regularization not applied to the bias term
        G_reg = G + lam * np.concatenate((np.zeros((1, 1)), w[1:]), axis=0)
        d = np.dot(np.linalg.inv(H_reg), (-1*G_reg))
        w = w + d
        check = np.sum(np.abs(d))
    return w

# Create empty list to store error rate for different lambda
train_err = list()
test_err = list()

# Get and store the error rate for training and testing set using lambda from 1 to 100
for i in range(len(lambda_array)):
    w_final = newton_method(x_train_bias, x_test_bias, lambda_array[i])
    prob_train = 1 / (1 + np.exp(-(np.dot(x_train_bias, w_final))))
    train_err.append((1 - np.sum((prob_train >= 0.5) == ytrain) / len(ytrain)) * 100)
    prob_test = 1 / (1 + np.exp(-(np.dot(x_test_bias, w_final))))
    test_err.append((1 - np.sum((prob_test >= 0.5) == ytest) / len(ytest)) * 100 )

# Transform the error rate list to array
train_err_array = np.array(train_err)
test_err_array = np.array(test_err)

# Ploting error rate versus lambda for training and testing set
plt.plot(lambda_array, train_err, label = 'Training Set')
plt.plot(lambda_array, test_err, label = 'Test Set')
plt.xlabel('Lambda')
plt.ylabel('Error rate %')
plt.title('Training and Testing Data Error rates')
plt.legend()
plt.show()

# Print error rate for lambda = 1, 10 and 100
for i in [0, 9, 27]:
    print('lambda =', lambda_array[i])
    print('training error:', train_err[i])
    print('testing error:', test_err[i])
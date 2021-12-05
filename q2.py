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


class GaussianClassifier():

    def __init__(self, x_1, y_1, x_2, y_2):

        self.log_xtrain = x_1
        self.ytrain = y_1
        self.log_xtest = x_2
        self.ytest = y_2

    # ML estimation of mean
    def mean(self, xn):
        return sum(xn) / len(xn)

    # ML estimation of variance
    def vari(self, xn, mu):
        return sum([(i - mu) ** 2 for i in xn]) / len(xn)

    def gaussian_fcn(self, x, mean, sig2):
        """
        Calculates the pdf of Normal(mean,sig2) distribution
        x: actual observation value
        mean: mean of observations
        sig2: variance of observations
        """

        return (1 / ((2 * np.pi * (sig2)) ** 0.5)) * np.exp((-0.5 * ((x - mean) ** 2)) / (sig2))

    def probmatrix_train(self, x_train, y_train):
        """
        Calculate feature values based on train data
        and store into separate lists
        """
        posmu_list = []
        posvari_list = []
        negmu_list = []
        negvari_list = []

        for col_idx in range(0, x_train.shape[1]):
            posmu = self.mean(x_train.loc[y_train[0] == 1, col_idx])
            posvari = self.vari(x_train.loc[y_train[0] == 1, col_idx], posmu)
            negmu = self.mean(x_train.loc[y_train[0] == 0, col_idx])
            negvari = self.vari(x_train.loc[y_train[0] == 0, col_idx], negmu)
            posmu_list.append(posmu)
            posvari_list.append(posvari)
            negmu_list.append(negmu)
            negvari_list.append(negvari)

        return posmu_list, posvari_list, negmu_list, negvari_list

    def predfcn(self, x_to_pred,
                pos_mean_list, pos_var_list,
                neg_mean_list, neg_var_list,
                pos_class_mle, neg_class_mle
                ):

        pred = []

        # loop through each row
        for row_idx in range(0, x_to_pred.shape[0]):

            row = x_to_pred.loc[row_idx, :]
            pos_p = np.log(pos_class_mle[0])
            neg_p = np.log(neg_class_mle[0])

            # loop through each feature
            for col_idx in range(0, x_to_pred.shape[1]):
                x_val = row[col_idx]
                pos_p += np.log(self.gaussian_fcn(x_val, pos_mean_list[col_idx], pos_var_list[col_idx]))
                neg_p += np.log(self.gaussian_fcn(x_val, neg_mean_list[col_idx], neg_var_list[col_idx]))

            # append prediction
            if pos_p > neg_p:
                pred.append(1)
            else:
                pred.append(0)

        return pred

    # error counts
    def error_fcn(self, pred, actual):
        return (sum([1 if int(p) != int(a) else 0 for p, a in zip(pred, actual)]) / len(pred)) * 100


    def run(self):

        # y ml estimate
        pos_class_mle = self.ytrain.sum() / len(self.ytrain)
        neg_class_mle = 1 - pos_class_mle
        # feature calculations
        posmean_list, posvar_list, negmean_list, negvar_list = self.probmatrix_train(self.log_xtrain, self.ytrain)
        # get predictions
        predtrain = self.predfcn(self.log_xtrain, posmean_list, posvar_list, negmean_list, negvar_list, pos_class_mle,
                                 neg_class_mle)
        predtest = self.predfcn(self.log_xtest, posmean_list, posvar_list, negmean_list, negvar_list, pos_class_mle,
                                neg_class_mle)
        # score and return results
        return self.error_fcn(predtrain, self.ytrain[0]), self.error_fcn(predtest, self.ytest[0])


gsc = GaussianClassifier(log_xtrain, ytrain, log_xtest, ytest)
gsc.run()


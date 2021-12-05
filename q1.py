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


class BetaBNClf():

    def __init__(self, a, b, x_1, y_1, x_2, y_2):

        # a : alpha parameter
        # b : beta parameter, equal to alpha B(a,a)
        # x_1: train data features
        # y_1: train true labels
        # x_2: test data features
        # y_2: test true labels

        self.a = a
        self.b = b
        self.bin_xtrain = x_1
        self.ytrain = y_1
        self.bin_xtest = x_2
        self.ytest = y_2

    # Beta function
    def calculate_beta_pdf(self, pos, n):
        """
        Calculates the pdf of Beta(a,b) distribution
        pos : number of positive instances
        neg: number of negative instances
        """
        return (pos + self.a) / (n + self.a + self.b)

    # calculate posterior probability p of each feature(each column)
    def calculate_beta_pdf_per_feature(self, xtr, ytr, j, cls):
        """
        Calculate one feature value per y_class
        """
        # j : the column_idx of xtrain data, i.e feacture name.
        # return two values. posprob : postive probability of jth feature
        # posprob : negative probability of jth feature
        n = list(ytr[0]).count(cls)
        conx = pd.concat([xtr[j], ytr[0]], axis=1)
        posc = conx.loc[(xtr[j] == 1) & (ytr[0] == cls), j]
        pos = len(posc)
        posprob = (pos + self.a) / (n + self.a + self.b)
        negprob = 1 - posprob

        return posprob, negprob

    def create_cls_data_from_train(self, x_train, y_train, y_class):
        """
        Calculate all feature values based on train data per y_class
        and store into dataframe
        """
        cls_data = pd.DataFrame()

        for col_idx in range(0, x_train.shape[1]):
            cls_data[col_idx] = list(self.calculate_beta_pdf_per_feature(
                xtr=x_train, ytr=y_train, j=col_idx, cls=y_class
            ))

        cls_data.index = [1, 0]

        return cls_data

    def predict(self, x_to_pred, pos_cls_data, neg_cls_data, pos_class_mle, neg_class_mle):
        """
        Predict on new data based on calculate feature information by class
        """

        pred = []

        # loop through each row
        for row_idx in range(0, x_to_pred.shape[0]):

            row = x_to_pred.loc[row_idx, :]
            pos_p = pos_class_mle[0]
            neg_p = neg_class_mle[0]

            # loop through each feature
            for col_idx in range(0, x_to_pred.shape[1]):
                x_val = row[col_idx]
                pos_p *= pos_cls_data.loc[x_val, col_idx]
                neg_p *= neg_cls_data.loc[x_val, col_idx]

            # compare and get the prediction
            if pos_p > neg_p:
                pred.append(1)
            else:
                pred.append(0)

        return pred
    # error rate calculate

    def calculate_error(self, pred, actual):
        """
        Calculate percentage of wrong classification
        """
        return (sum([1 if int(p) != int(a) else 0 for p, a in zip(pred, actual)]) / len(pred)) * 100

    def run(self):
        """
        Run pipeline
        """
        # class mle
        pos_class_mle = self.ytrain.sum() / len(self.ytrain)
        neg_class_mle = 1 - pos_class_mle
        # feature calculations
        pos_cls_data = self.create_cls_data_from_train(self.bin_xtrain, self.ytrain, 1)
        neg_cls_data = self.create_cls_data_from_train(self.bin_xtrain, self.ytrain, 0)
        # get predictions
        train_pred = self.predict(self.bin_xtrain, pos_cls_data, neg_cls_data, pos_class_mle, neg_class_mle)
        test_pred = self.predict(self.bin_xtest, pos_cls_data, neg_cls_data, pos_class_mle, neg_class_mle)
        # score and return results
        return self.calculate_error(train_pred, self.ytrain[0]), \
               self.calculate_error(test_pred, self.ytest[0])


# initialise empty list for storage
list_of_alphas = list(np.arange(0, 100.5, 0.5))
train_error_list = []
test_error_list = []

# loop over alpha values
for set_a in list_of_alphas:
    bbc = BetaBNClf(set_a, set_a, bin_xtrain, ytrain, bin_xtest, ytest)
    train_error, test_error = bbc.run()
    train_error_list.append(train_error)
    test_error_list.append(test_error)

# plot chart
plt.plot(list_of_alphas, train_error_list, label='Train Error')
plt.plot(list_of_alphas, test_error_list, label='Test Error')
plt.ylabel('Error Rate %')
plt.xlabel('Alpha Value')
plt.title('Training and Testing Data Error Rates')
plt.legend()
plt.show()

for val in [1,10,100]:
    idx = list_of_alphas.index(val)
    print('The training error =', str(train_error_list[idx])+'%')
    print('The testing error =',str(test_error_list[idx])+'%')
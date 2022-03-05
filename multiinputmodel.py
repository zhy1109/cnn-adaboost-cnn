
# import keras
import numpy as np
# import matplotlib.pyplot as plt
# from matplotlib.pyplot import imshow
# import pandas as pd
from tensorflow.keras import optimizers
import keras_metrics as km
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from numpy.core.umath_tests import inner1d
# from copy import deepcopy
# import keras_metrics as km
from keras.callbacks import TensorBoard
##kerase & CNN:
# from keras import models as Models
# from keras.models import Sequential
from sklearn.preprocessing import OneHotEncoder  # LabelBinarizer
# from keras.callbacks import TensorBoard
# import keras.backend as K
# from sklearn.metrics import precision_score, recall_score, f1_score

class AdaBoostClassifiermulti(object):
    '''
    Parameters
    -----------
    base_estimator: object
        The base model from which the boosted ensemble is built.

    n_estimators: integer, optional(default=50)
        The maximum number of estimators

    learning_rate: float, optional(default=1)

    algorithm: {'SAMME','SAMME.R'}, optional(default='SAMME.R')
        SAMME.R uses predicted probabilities to update wights, while SAMME uses class error rate

    random_state: int or None, optional(default=None)


    Attributes
    -------------
    estimators_: list of base estimators

    estimator_weights_: array of floats
        Weights for each base_estimator

    estimator_errors_: array of floats
        Classification error for each estimator in the boosted ensemble.

    Reference:
    1. [multi-adaboost](https://web.stanford.edu/~hastie/Papers/samme.pdf)

    2. [scikit-learn:weight_boosting](https://github.com/scikit-learn/
    scikit-learn/blob/51a765a/sklearn/ensemble/weight_boosting.py#L289)

    '''

    def __init__(self, *args, **kwargs):
        # if kwargs and args:
        #     raise ValueError(
        #         '''AdaBoostClassifier can only be called with keyword
        #            arguments for the following keywords: base_estimator ,n_estimators,
        #             learning_rate,algorithm,random_state''')
        # allowed_keys = ['base_estimator', 'n_estimators', 'learning_rate', 'algorithm', 'random_state', 'epochs']
        # keywords_used = kwargs.keys()
        # for keyword in keywords_used:
        #     if keyword not in allowed_keys:
        #         raise ValueError(keyword + ":  Wrong keyword used --- check spelling")
        # n_estimators = 50
        # learning_rate = 0.03
        # algorithm = 'SAMME.R'
        random_state = None
        if kwargs and not args:
            if 'base_estimator' in kwargs:
                base_estimator = kwargs.pop('base_estimator')
            else:
                raise ValueError('''base_estimator can not be None''')
            if 'n_estimators' in kwargs: n_estimators = kwargs.pop('n_estimators')
            if 'learning_rate' in kwargs: learning_rate = kwargs.pop('learning_rate')
            if 'algorithm' in kwargs: algorithm = kwargs.pop('algorithm')
            if 'random_state' in kwargs: random_state = kwargs.pop('random_state')
            if 'epochs' in kwargs: epochs = kwargs.pop('epochs')
            if 'probacoefficient' in kwargs: probacoefficient = kwargs.pop('probacoefficient')
            if 'imbalance' in kwargs: imbalance = kwargs.pop('imbalance')
        self.base_estimator_ = base_estimator
        self.n_estimators_ = n_estimators
        self.learning_rate_ = learning_rate
        self.algorithm_ = algorithm
        self.random_state_ = random_state
        self.estimators_ = list()
        self.estimator_weights_ = np.zeros(self.n_estimators_)
        self.estimator_errors_ = np.ones(self.n_estimators_)
        self.epochs = epochs
        self.probacoefficient = probacoefficient
        self.imbalance = imbalance

    def _samme_proba(self, estimator, n_classes, X,weight):
        """Calculate algorithm 4, step 2, equation c) of Zhu et al [1].
        References
        .. [1] J. Zhu, H. Zou, S. Rosset, T. Hastie, "Multi-class AdaBoost", 2009.
        """
        # print(len(X),X[0].shape,'220',self.batch_size,'99')
        proba = estimator.predict(X)*weight#,steps=16
        # Displace zero probabilities so the log is defined.
        # Also fix negative elements which may occur with
        # negative sample weights.
        proba[proba < np.finfo(proba.dtype).eps] = np.finfo(proba.dtype).eps
        log_proba = np.log(proba)
        probasum=(n_classes - 1) * (log_proba - (1. / n_classes)
                                   * log_proba.sum(axis=1)[:, np.newaxis])
        return probasum

    def fit(self, X, y, batch_size):
        self.batch_size = batch_size
        ## CNN:
        #        self.epochs = epochs
        self.n_samples = X[0].shape[0]
        # self.n_samples = len(X)
        # There is hidden trouble for classes, here the classes will be sorted.
        # So in boost we have to ensure that the predict results have the same classes sort
        self.classes_ = np.array(sorted(list(set(y))))
        ############for CNN (2):
        #        yl = np.argmax(y)
        #        self.classes_ = np.array(sorted(list(set(yl))))
        self.n_classes_ = len(self.classes_)
        for iboost in range(self.n_estimators_):
            if iboost == 0:
                sample_weight = np.ones(self.n_samples) / self.n_samples
                a=np.where(y == 1)
                sample_weight[a]=sample_weight[a]+self.imbalance*sample_weight[0]
            sample_weight, estimator_weight, estimator_error= self.boost(X, y, sample_weight)
            if estimator_error == None:
                break
            # append error and weight
            self.estimator_errors_[iboost] = estimator_error
            self.estimator_weights_[iboost] = estimator_weight
            # mask = self.estimator_weights_!=0
            # self.estimator_weights_= self.estimator_weights_[mask]  # [135  30 125]
            if estimator_error <= 0:
                break
        # mask = self.estimator_weights_!=0
        # self.estimator_weights_= self.estimator_weights_[mask]
        return self

    def boost(self, X, y, sample_weight):
        if self.algorithm_ == 'SAMME':
            return self.discrete_boost(X, y, sample_weight)
        elif self.algorithm_ == 'SAMME.R':
            return self.real_boost(X, y, sample_weight)

    def real_boost(self, X,y, sample_weight):
        if len(self.estimators_) == 0:
            estimator = self.deepcopy_CNN(self.base_estimator_)  # deepcopy of self.base_estimator_ Transfer learning
        else:
            estimator = self.deepcopy_CNN(self.estimators_[-1])  # deepcopy CNN
        if self.random_state_:
                estimator.set_params(random_state=1)
        # if self.random_state_:
        #    estimator.set_params(random_state=1)
        lb = OneHotEncoder(sparse=False)
        y_b = y.reshape(len(y), 1)
        y_b = lb.fit_transform(y_b)
        y_b = np.array(y_b)
        if len(self.estimators_) == 0:
            estimator.fit(X, y_b,epochs=25,batch_size = 32,sample_weight=sample_weight,shuffle=False,validation_split=0.01)#,callbacks=[TensorBoard(log_dir='./log')]
        else:
            estimator.fit(X, y_b, epochs=15, batch_size=32, sample_weight=sample_weight, shuffle=False,validation_split=0.01)
        # estimator.fit([X,X], y_b, sample_weight=sample_weight, validation_split=0.1, epochs=self.epochs,callbacks=[TensorBoard(log_dir='./log1')]
        #               batch_size=self.batch_size, callbacks=[TensorBoard(log_dir='./log1')])#training process
        ############################################################
        y_pred = estimator.predict(X)#,steps=16
        y_pred_l = np.argmax(y_pred, axis=1)
        incorrect = y_pred_l != y
        #########################################################
        estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)
        # estimator_error = np.mean(estimator_error)
        # if worse than random guess, stop boosting
        #print(type(estimator_error))
        if estimator_error >= 1.0 - 1 / self.n_classes_:
            return None, None, None
        if estimator_error == 0 :
            estimator_error=0.0001
        ###new add
        estimator_weight = self.learning_rate_ * (
                np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

        if estimator_weight <= 0:
            return None, None, None
        y_predict_proba=y_pred
        # repalce zero    ~.~
        y_predict_proba[y_predict_proba < np.finfo(y_predict_proba.dtype).eps] = np.finfo(y_predict_proba.dtype).eps

        y_codes = np.array([-1. / (self.n_classes_ - 1), 1.])
        y_coding = y_codes.take(self.classes_ == y[:, np.newaxis])
        # for sample weight update
        # intermediate_variable = (-1. * self.learning_rate_ * (((self.n_classes_ - 1) / self.n_classes_) *
        #                                                       inner1d(y_coding, np.log(
        #                                                           y_predict_proba))))  # dot iterate for each row
        # intermediate_variable = intermediate_variable * self.imbalance
        intermediate_variable = (-1. * self.learning_rate_ * ((inner1d(y_coding, np.log( y_predict_proba)))))
        # update sample weight

        sample_weight *= np.exp(intermediate_variable)

        # print(sample_weight)
        sample_weight_sum = np.sum(sample_weight, axis=0)
        if sample_weight_sum <= 0:
            return None, None, None
        # normalize sample weight
        sample_weight /= sample_weight_sum
        # append the estimator
        # if index>0.8:
        self.estimators_.append(estimator)
        # else:
        #     estimator_weight=0
        return sample_weight, estimator_weight, estimator_error

    def discrete_boost(self, X, y, sample_weight):
            #        estimator = deepcopy(self.base_estimator_)
            ############################################### my code:

            # if len(self.estimators_) == 0:
            #     # Copy CNN to estimator:
            #     estimator = self.deepcopy_CNN(self.base_estimator_)  # deepcopy of self.base_estimator_
            # else:
            #     # estimator = deepcopy(self.estimators_[-1])
            #     estimator = self.deepcopy_CNN(self.estimators_[-1])  # deepcopy CNN
            #     ###################################################
            #
            # if self.random_state_:
            #     estimator.set_params(random_state=1)
            if len(self.estimators_) == 0:
                estimator = self.deepcopy_CNN(self.base_estimator_)  # deepcopy of self.base_estimator_
            else:
                estimator = self.deepcopy_CNN(self.estimators_[-1])  # deepcopy CNN
            if self.random_state_:
                estimator.set_params(random_state=1)
            #        estimator.fit(X, y, sample_weight=sample_weight)
            #################################### CNN (3) binery label:
            # lb=LabelBinarizer()
            # y_b = lb.fit_transform(y)
            lb = OneHotEncoder(sparse=False)
            y_b = y.reshape(len(y), 1)
            y_b = lb.fit_transform(y_b)
            estimator.fit(X, y_b, sample_weight=sample_weight, epochs=self.epochs, shuffle=False,batch_size=self.batch_size,validation_split=0.3)#
            # estimator.fit(X, y_b, sample_weight=sample_weight, epochs = self.epochs, batch_size = self.batch_size)
            ############################################################
            y_pred = estimator.predict(X)
            # incorrect = y_pred != y
            ############################################ (4) CNN :
            y_pred_l = np.argmax(y_pred, axis=1)
            incorrect = y_pred_l != y
            #######################################################
            estimator_error = np.dot(incorrect, sample_weight) / np.sum(sample_weight, axis=0)

            # if worse than random guess, stop boosting
            if estimator_error >= 1 - 1 / self.n_classes_:
                return None, None, None
            # update estimator_weight
            #        estimator_weight = self.learning_rate_ * np.log((1 - estimator_error) / estimator_error) + np.log(
            #            self.n_classes_ - 1)
            estimator_weight = self.learning_rate_ * (
                        np.log((1. - estimator_error) / estimator_error) + np.log(self.n_classes_ - 1.))

            if estimator_weight <= 0:
                return None, None, None
            # update sample weight
            sample_weight *= np.exp(estimator_weight * incorrect)
            sample_weight_sum = np.sum(sample_weight, axis=0)
            if sample_weight_sum <= 0:
                return None, None, None

            # normalize sample weight
            sample_weight /= sample_weight_sum

            # append the estimator
            self.estimators_.append(estimator)

            return sample_weight, estimator_weight, estimator_error
    def deepcopy_CNN(self, base_estimator0):
        #Copy CNN (self.base_estimator_) to estimator:
        # config=base_estimator0.get_config()
        estimator = tf.keras.Model().from_config(base_estimator0.get_config())
        # estimator = Sequential.from_config(config)
        weights = base_estimator0.get_weights()
        estimator.set_weights(weights)
        def lossall(estimator):
            def binary_loss(y_true, y_pred):
                loss1 = tf.keras.losses.binary_crossentropy(y_true, y_pred)
                return loss1
            return binary_loss
            # loss1 = tf.keras.losses.BinaryCrossentropy
            fea1 = model.get_layer('conv2d').output
            fea2 = model.get_layer('conv2d_4').output
            fea3 = model.get_layer('conv2d_1').output
            fea4 = model.get_layer('conv2d_5').output
            fea5 = model.get_layer('conv2d_2').output
            fea6 = model.get_layer('conv2d_6').output
            lossconv1 = tf.square(
                MMD(tf.reshape(fea1[:, :, :, 1], [15, 39]), tf.reshape(fea2[:, :, :, 1], [15, 39]), beta=1))
            lossconv2 = tf.square(
                MMD(tf.reshape(fea3[:, :, :, 1], [7, 19]), tf.reshape(fea4[:, :, :, 1], [7, 19]), beta=1))
            lossconv3 = tf.square(
                MMD(tf.reshape(fea5[:, :, :, 1], [3, 9]), tf.reshape(fea6[:, :, :, 1], [3, 9]), beta=1))
            loss = 1*lossconv1/1 + 1*lossconv2/1 + 1*lossconv3/1 + loss1
            return loss
        estimator.compile(loss=[lossall(estimator)], optimizer='adam', metrics=[km.f1_score(), km.precision(), km.recall()])#loss='BinaryCrossentropy',loss=[lossall(estimator)]
        return estimator
    def predict(self, X):
        n_classes = self.n_classes_
        classes = self.classes_[:, np.newaxis]
        # pred = None
        if self.algorithm_ == 'SAMME.R':
            # The weights are all 1. for SAMME.R
            # pred=sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
            pred = sum(self._samme_proba(estimator, n_classes, X,weight) for estimator,weight in zip(self.estimators_,self.estimator_weights_))
            # pred=(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
            # pred1=sum(weight*self._samme_proba(estimator, n_classes, X) for estimator,weight in self.estimators_,self.estimator_weights_)
            # pred1=pred
            # pred = sum(self._samme_proba(estimator, n_classes, X) for estimator in self.estimators_)
        else:  # self.algorithm == "SAMME"
                       # pred = sum((estimator.predict(X) == classes).T * w
                       #            for estimator, w in zip(self.estimators_,
                       #                                    self.estimator_weights_))
            ########################################CNN disc
            pred = sum((estimator.predict(X).argmax(axis=1) == classes).T * w#,steps=16
                       for estimator, w in zip(self.estimators_,
                                               self.estimator_weights_))
        pred /= self.estimator_weights_.sum()
        if n_classes == 2:
            pred[:, 0] *= -1
            pred = pred.sum(axis=1)
            return self.classes_.take(pred > 0, axis=0)
        return self.classes_.take(np.argmax(pred, axis=1), axis=0)
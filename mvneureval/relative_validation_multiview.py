#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

import numpy as np
from sklearn.metrics import zero_one_loss
from mvneureval.utils import kuhn_munkres_algorithm
import logging
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import math

logging.basicConfig(format='%(asctime)s, %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.INFO)


class RelativeValidationMultiview:
    """
    This class allows to perform the relative clustering validation procedure.
    A supervised algorithm is required to test cluster stability.
    Labels output from a clustering algorithm are used as true labels.

    :param s: initialized class for the supervised method.
    :type s: class
    :param c: initialized class for clustering algorithm.
    :type c: class
    :param preprocessing: initialized class for preprocessing method.
    :type preprocessing: class
    :param nrand: number of iterations to normalize cluster stability.
    :type nrand: int
    """

    def __init__(self, c, s, nrand=10):
        """
        Construct method.
        """
        self.clust_method = c
        self.class_method = s
        self.nrand = nrand
        
    def GLMcorrection_multimodal(self, modalities,covariates, tr_idx, val_idx):
        """
        Eliminate the confound of covariate, such as age and sex, from the input features.
        :param X_train: array, training features
        :param Y_train: array, training labels
        :param covar_train: array, training covariate data
        :param X_test: array, test labels
        :param covar_test: array, test covariate data
        :param num_col_DTI: index of dataset column corresponding to white matter features (i.e., "ACR" column)
        :return: corrected training & test feature data
        
        """
        X_train_dic = {mod:None for mod in modalities}
        X_test_dic = {mod:None for mod in modalities}
        cov_train_dic = {mod:None for mod in modalities}
        cov_test_dic = {mod:None for mod in modalities}
            
        for mod in modalities:
            X_train_dic[mod] = np.array(modalities[mod])[tr_idx]
            if X_train_dic[mod].ndim==1:
                X_train_dic[mod] = X_train_dic[mod].reshape(-1,1)
            X_test_dic[mod]= np.array(modalities[mod])[val_idx]
            if X_test_dic[mod].ndim==1:
                X_test_dic[mod] = X_test_dic[mod].reshape(-1,1)
            cov_train_dic[mod] = np.array(covariates[mod])[tr_idx]
            if cov_train_dic[mod].ndim==1:
                cov_train_dic[mod] = cov_train_dic[mod].reshape(-1,1)
            cov_test_dic[mod]= np.array(covariates[mod])[val_idx]
            if cov_test_dic[mod].ndim==1:
                cov_test_dic[mod] = cov_test_dic[mod].reshape(-1,1)
        
        
        # normalize the covariate z-scoring
        train_scaled_dic = {mod:None for mod in modalities}
        test_scaled_dic = {mod:None for mod in modalities}
        trcov_scaled_dic = {mod:None for mod in modalities}
        tscov_scaled_dic = {mod:None for mod in modalities}
        scaler_data = StandardScaler()
        scaler_cov = StandardScaler()
        
        for mod in modalities:
            train_scaled_dic[mod] = scaler_data.fit_transform(X_train_dic[mod])
            test_scaled_dic[mod] = scaler_data.transform(X_test_dic[mod])
            trcov_scaled_dic[mod]= scaler_cov.fit_transform(cov_train_dic[mod])
            tscov_scaled_dic[mod]= scaler_cov.transform(cov_test_dic[mod])
    
        # Adjust data for confounds of covariate
        train_cor_dic = {mod:None for mod in modalities}
        test_cor_dic = {mod:None for mod in modalities}
        
        for mod in modalities:
            beta = np.linalg.lstsq(trcov_scaled_dic[mod], train_scaled_dic[mod], rcond=None)
            train_cor_dic[mod] = (train_scaled_dic[mod].T - beta[0].T @ trcov_scaled_dic[mod].T).T
            test_cor_dic[mod] = (test_scaled_dic[mod].T - beta[0].T @ tscov_scaled_dic[mod].T).T
        
        return train_cor_dic, test_cor_dic

    def train(self, train_cor_dic, tr_lab=None):
        """
        Method that performs training. It compares the clustering labels on training set
        (i.e., A(X) computed by :class:`neureval.relative_validation_confounds.RelativeValidationConfounds.clust_method`) against
        the labels obtained from the classification algorithm
        (i.e., f(X), computed by :class:`neureval.relative_validation_confounds.RelativeValidationConfounds.class_method`).
        It returns the misclassification error, the supervised model fitted to the data,
        and both clustering and classification labels.

        :param X_train_cor: counfound corrected training dataset.
        :type X_train_cor: ndarray, (n_samples, n_features)
        :param tr_lab: cluster labels found during CV for clustering methods with no `n_clusters` parameter.
            If not None the clustering method is not performed on the whole test set. Default None.
        :type tr_lab: list
        :return: misclassification error, fitted supervised model object, clustering and classification labels.
        :rtype: float, object, ndarray (n_samples,)
        """
        mv_tr_data = [np.array(train_cor_dic[mod]) for mod in train_cor_dic]
        
        if tr_lab is None:
            clustlab_tr = self.clust_method.fit_predict(mv_tr_data)  # A_k(X)
            mv_tr_embedding = self.clust_method.embedding_
        else:
            clustlab_tr = tr_lab
            fitcluster_tr = self.clust_method.fit(mv_tr_data)
            mv_tr_embedding = self.clust_method.embedding_
        if len([cl for cl in clustlab_tr if cl >= 0]) == 0:
            logging.info(f"No clusters found during training with {self.clust_method}.")
            return None
        
        fitclass_tr = self.class_method.fit(mv_tr_embedding, clustlab_tr)
        classlab_tr = fitclass_tr.predict(mv_tr_embedding)
        misclass = zero_one_loss(clustlab_tr, classlab_tr)
        
        return misclass, fitclass_tr, clustlab_tr, mv_tr_embedding
        

    def test(self, test_cor_dic, fit_model):
        """
        Method that compares test set clustering labels (i.e., A(X'), computed by
        :class:`neureval.relative_validation_confounds.RelativeValidationConfounds.clust_method`) against
        the (permuted) labels obtained through the classification algorithm fitted to the training set
        (i.e., f(X'), computed by
        :class:`reval.relative_validation.RelativeValidationConfounds.class_method`).
        It returns the misclassification error, together with
        both clustering and classification labels.

        :param X_test_cor: confound corrected test dataset.
        :type X_test_cor: ndarray, (n_samples, n_features)
        :param fit_model: fitted supervised model.
        :type fit_model: class
        :param fit_preproc: fitted preprocessing method.
        :type fit_preproc: class
        :return: misclassification error, clustering and classification labels.
        :rtype: float, dictionary of ndarrays (n_samples,)
        """
        mv_ts_data = [np.array(test_cor_dic[mod]) for mod in test_cor_dic]
        
        clustlab_ts = self.clust_method.fit_predict(mv_ts_data)  # A_k(X')
        if len([cl for cl in clustlab_ts if cl >= 0]) == 0:
            logging.info(f"No clusters found during testing with {self.clust_method}")
            return None
        
        mv_ts_embedding = self.clust_method.embedding_
        classlab_ts = fit_model.predict(mv_ts_embedding)
        bestperm = kuhn_munkres_algorithm(np.int32(classlab_ts), np.int32(clustlab_ts))  # array of integers
        misclass = zero_one_loss(classlab_ts, bestperm)
        
        return misclass, bestperm, mv_ts_embedding

    def rndlabels_traineval(self, mv_tr_embedding, mv_ts_embedding, train_labels, test_labels):
        """
        Method that performs random labeling on the training set
        (N times according to
        :class:`neureval.relative_validation_confounds.RelativeValidationConfounds.nrand` instance attribute) and evaluates
        the fitted models on test set.

        :param X_train_cor: confound corrected training dataset.
        :type X_train_cor: ndarray, (n_samples, n_features)
        :param X_test_cor: confound corrected test dataset.
        :type X_test_cor: ndarray, (n_samples, n_features)
        :param train_labels: training set clustering labels.
        :type train_labels: ndarray, (n_samples,)
        :param test_labels: test set clustering labels.
        :type test_labels: ndarray, (n_samples,)
        :return: averaged misclassification error on the test set.
        :rtype: float
        """
        np.random.seed(0)
        shuf_tr = [np.random.permutation(train_labels)
                   for _ in range(self.nrand)]
        misclass_ts = list(map(lambda x: self._rescale_score_(mv_tr_embedding, mv_ts_embedding, x, test_labels), shuf_tr))
        return np.mean(misclass_ts)

    def _rescale_score_(self, xtr, xts, randlabtr, labts):
        """
        Private method that computes the misclassification error when predicting test labels
        with classification model fitted on training set with random labels.

        :param xtr: training dataset.
        :type xtr: ndarray, (n_samples, n_features)
        :param xts: test dataset.
        :type xts: ndarray, (n_samples, n_features)
        :param randlabtr: random labels.
        :type randlabtr: ndarray, (n_samples,)
        :param labts: test set labels.
        :type labts: ndarray, (n_samples,)
        :return: misclassification error.
        :rtype: float
        """
        self.class_method.fit(xtr, randlabtr)
        pred_lab = self.class_method.predict(xts)
        me_ts = zero_one_loss(pred_lab, kuhn_munkres_algorithm(np.int32(pred_lab), np.int32(labts)))
        return me_ts

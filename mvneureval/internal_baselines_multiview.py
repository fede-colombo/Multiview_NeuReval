"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""

import numpy as np
from sklearn.preprocessing import StandardScaler
import math

def select_best(data, modalities, covariates, c, int_measure, select='max', nclust_range=None):
    """
    Select the best number of clusters that minimizes/maximizes
    the internal measure selected.

    :param data: dataset.
    :type data: array-like
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing: data reduction algorithm class, default None.
    :type preprocessing: obj
    :param select: it can be 'min', if the internal measure is to be minimized
        or 'max' if the internal measure should be macimized.
    :type select: str
    :param nclust_range: Range of clusters to consider, default None.
    :type nclust_range: list
    :param combined_data: define whether multimodal data are used as input features. 
        If True, different sets of covariates will be applied for each modality
        e.g. correction for TIV only for grey matter features. Default False
    :type combined_data: boolean value
    :return: internal score and best number of clusters.
    :rtype: float, int
    """
    X_dic = {mod:None for mod in modalities}
    cov_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       X_dic[mod] = np.array(modalities[mod])
       cov_dic[mod] = np.array(covariates[mod])
       
    # normalize the covariate z-scoring
    X_scaled_dic = {mod:None for mod in modalities}
    cov_scaled_dic = {mod:None for mod in modalities}
    scaler = StandardScaler()
   
    for mod in modalities:
       X_scaled_dic[mod] = scaler.fit_transform(X_dic[mod])
       cov_scaled_dic[mod]= scaler.fit_transform(cov_dic[mod])
   
   
    # Adjust data for confounds of covariate
    X_cor_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       beta = np.linalg.lstsq(cov_scaled_dic[mod], X_scaled_dic[mod], rcond=None)
       X_cor_dic[mod] = (X_scaled_dic[mod].T - beta[0].T @ cov_scaled_dic[mod].T).T
    
    #mv_data = np.concatenate([X_cor_dic[mod]for mod in X_cor_dic], axis=1)
    mv_data = [np.array(X_cor_dic[mod]) for mod in X_cor_dic]
        
    if nclust_range is not None:
        scores = []
        label_vect = []
        for ncl in nclust_range:
            if 'n_clusters' in c.get_params().keys():
                c.n_clusters = ncl
            else:
                c.n_components = ncl
            label = c.fit_predict(mv_data)
            mv_embedding = c.embedding_
            scores.append(int_measure(mv_embedding, label))
            label_vect.append(label)
    else:
        label = c.fit_predict(mv_data)
        mv_embedding = c.embedding_
        scores.append(int_measure(mv_embedding, label))
        return best, len([lab for lab in np.unique(label) if lab >= 0]), label
        
    if select == 'max':
        best = np.where(np.array(scores) == max(scores))[0]
    elif select == 'min':
        best = np.where(np.array(scores) == min(scores))[0]
    if len(set(label_vect[int(max(best))])) == nclust_range[int(max(best))]:
        return scores[int(max(best))], nclust_range[int(max(best))], label_vect[int(max(best))]
    else:
        return scores[int(max(best))], len(set(label_vect[int(max(best))])), label_vect[int(max(best))]


def evaluate_best(data, modalities, covariates, c, int_measure, ncl=None):
    """
    Function that, given a number of clusters, returns the corresponding internal measure
    for a dataset.

    :param data: dataset.
    :type data: array-like
    :param c: clustering algorithm class.
    :type c: obj
    :param int_measure: internal measure function.
    :type int_measure: obj
    :param preprocessing:  dimensionality reduction algorithm class, default None.
    :type preprocessing: obj
    :param ncl: number of clusters.
    :type ncl: int
    :param combined_data: define whether multimodal data are used as input features. 
        If True, different sets of covariates will be applied for each modality
        e.g. correction for TIV only for grey matter features. Default False
    :type combined_data: boolean value
    :return: internal score.
    :rtype: float
    """
    X_dic = {mod:None for mod in modalities}
    cov_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       X_dic[mod] = np.array(modalities[mod])
       cov_dic[mod] = np.array(covariates[mod])
       
    # normalize the covariate z-scoring
    X_scaled_dic = {mod:None for mod in modalities}
    cov_scaled_dic = {mod:None for mod in modalities}
    scaler = StandardScaler()
   
    for mod in modalities:
       X_scaled_dic[mod] = scaler.fit_transform(X_dic[mod])
       cov_scaled_dic[mod]= scaler.fit_transform(cov_dic[mod])
   
   
    # Adjust data for confounds of covariate
    X_cor_dic = {mod:None for mod in modalities}
    
    for mod in modalities:
       beta = np.linalg.lstsq(cov_scaled_dic[mod], X_scaled_dic[mod], rcond=None)
       X_cor_dic[mod] = (X_scaled_dic[mod].T - beta[0].T @ cov_scaled_dic[mod].T).T

    mv_data = np.concatenate([X_cor_dic[mod]for mod in X_cor_dic], axis=1)
    
    if 'n_clusters' in c.get_params().keys():
        c.n_clusters = ncl
    else:
        c.n_components = ncl
        label = c.fit_predict(mv_data)
    
    return int_measure(mv_data, label)

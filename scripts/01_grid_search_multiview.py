# -*- coding: utf-8 -*-
"""
@author: Federica Colombo
         Psychiatry and Clinical Psychobiology Unit, Division of Neuroscience, 
         IRCCS San Raffaele Scientific Institute, Milan, Italy
"""
### GRID-SEARCH CROSS-VALIDATION MULTIVIEW CLUSTERING/CLASSIFIER PARAMETERS
# Example of NeurReval application with multiview spectral clustering as clustering algorithm and SVC as classifier #

# Make the required imports
import pandas as pd
import numpy as np
from mvneureval.param_selection_multiview import ParamSelectionMultiview
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import os
import pickle as pkl
import sys
from mvneureval.utils import kuhn_munkres_algorithm
import logging
from mvlearn.cluster import MultiviewSpectralClustering


## Define working directories and create a new folder called 'models'
main_path = 'path/to/working/directory'
data_path = os.path.join(main_path,'models')

# Define output folder called 'results'. 
# Within 'results, create a subfolder corresponding to the input features (e.g., 'GM + FA').
# Within this subfolder, create another folder corresponding to the set of covariates used (e.g., 'age_sex_TIV')
out_dir = os.path.join(main_path, 'results', 'feature_name', 'covariates')
os.makedirs(out_dir, exist_ok=True)
os.chdir(out_dir)


# Import data and covariates files
database = 'database_name.xlsx'
covariates_file = 'database_name.xlsx'
# If there are multiple sheets, specify the name of the current sheet 
data = pd.read_excel(os.path.join(data_path,database), sheet_name='sheet_name')
cov= pd.read_excel(os.path.join(data_path, covariates_file), sheet_name='sheet_name')

# Define two dictionaries for modalities (i.e., views in multiview clustering) and covariates variables.
# For each kind of modality, specify the indexes of the columuns related to the features to be used for clustering
# You can also specify different covaiates for each kind of modality
modalities = {'modality_name_01': data.iloc[:, 2:374],
              'modality_name_02': data.iloc[:,374:417],
              'modality_name_03': data.iloc[:,417:460]}

covariates = {'modality_name_01': cov.iloc[:, 2:],
              'modality_name_02': cov.iloc[:,2:-1],
              'modality_name_03': cov.iloc[:,2:-1]}

# Define multiview clustering and classifier parameters to be optimized.
# params should be a dictionary of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} 
# including the lists of classifiers and clustering methods to fit to the data.
params = {'s': {'C': [0.01, 0.1, 1, 10, 100, 1000],
                'kernel':['linear', 'rbf']},
          'c': {'affinity':['rbf', 'nearest_neighbors']}}

# Define classifier and multiview clustering methods
c = MultiviewSpectralClustering(random_state=42)
s = SVC(random_state=42)

# Run ParamSelectionMultiview that implements grid search cross-validation to select the best combinations of parameters for fixed classifier/clustering algorithms.
# Parameters to be specified:
# cv: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# iter_cv: number of repeated cross-validation, default 10
# clust_range: list with number of clusters (e.g., list(range(2,3))), default None
# strat: stratification vector for cross-validation splits, default None    
best_model = ParamSelectionMultiview(params, 
                                      cv=2, 
                                      s=s, 
                                      c=c,
                                      nrand=10,
                                      n_jobs=-1,
                                      iter_cv=1,
                                      clust_range=None,
                                      strat=None)

best_model.fit(data,modalities,covariates)

# Save model's parameters in the output directory. Change file name to match the model you performed
best_results = best_model.best_param_
pkl.dump(best_results, open('./best_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./best_results_multiview_spectral_SVM.pkl')

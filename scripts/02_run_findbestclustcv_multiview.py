# -*- coding: utf-8 -*-
"""
Created on Sat Mar  4 11:35:25 2023

@author: fcolo
"""

### RUN FindBestClustCVMultiview WITH OPTIMIZED MULTIVIEW CLUSTERING/CLASSIFIER PARAMETERS

# Make the required imports
import pandas as pd
import numpy as np
from mvneureval.best_nclust_cv_multiview import FindBestClustCVMultiview
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import AgglomerativeClustering, KMeans, DBSCAN, SpectralClustering
from sklearn.metrics import zero_one_loss, adjusted_mutual_info_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.mixture import GaussianMixture
import os
import pickle as pkl
import logging
import sys
from mvneureval.visualization import plot_metrics
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


# Define multiview clusteirng and classifier algortihms with optimized parameters
c = MultiviewSpectralClustering(affinity='nearest_neighbors',random_state=42)
s = SVC(C=0.1, kernel='rbf', random_state=42)

# Initialize FindBestClustCVMultiview class. It performs (repeated) k-folds cross-validation to select the best number of clusters.
# Parameters to be specified:
# nfold: cross-validation folds
# nrand: number of random labelling iterations, default 10
# n_jobs: number of jobs to run in parallel, default (number of cpus - 1)
# clust_range: list with number of clusters (e.g., list(range(2,3))), default None
findbestclust = FindBestClustCVMultiview(c,s, nfold=2, nrand=10,  n_jobs=-1, nclust_range=None)

# Run FindBestClustCVMultiview. It returns normlaized stability (metrics), best number of clusters (bestncl), and clusters' labels (tr_lab).
# Parameters to be specified:
# iter_cv: number of repeated cross-validation, default 1
# strat: stratification vector for cross-validation splits, default None
metrics, bestncl, tr_lab = findbestclust.best_nclust(data, modalities, covariates, iter_cv=1, strat_vect=None)
val_results = list(metrics['val'].values())
val_results = np.array(val_results, dtype=object)

print(f"Best number of clusters: {bestncl}")
print(f"Validation set normalized stability (misclassification): {metrics['val'][bestncl]}")
print(f"Result accuracy (on test set): "
      f"{1-val_results[0,0]}")

# Normalized stability plot. For each number of clusters, normalized stabilities are represented for both training (dashed line) and validation sets (continuous line).
# Colors for training and validation sets can be changed (default: ('black', 'black')).
# To save the plot, specify the file name for saving figure in png format.
plot = plot_metrics(metrics, color=('black', 'black'), save_fig='plot_model_name.png')

# Save database with cluster labels for post-hoc analyses
labels = pd.DataFrame(tr_lab, columns=['Clustering labels'])
data_all = pd.concat([data, labels], axis=1)
data_all.to_csv('Labels_model_name.csv', index=True)



### COMPUTE INTERNAL MEASURES FOR COMPARISON

# In case you want to also compute internal measures for comparison, make the following imports
from sklearn.metrics import adjusted_mutual_info_score, silhouette_score, davies_bouldin_score
from mvneureval.internal_baselines_multiview import select_best
from mvneureval.utils import kuhn_munkres_algorithm

# The code is organized in different sections for each kind of measures (i.e., silhouette score, Davies-Boudin score, AIC and BIC).
# If you want to calculate only some scores, comment the ones you don't need

# SILHOUETTE SCORE
# Use select_best to calcluate silhouette score. Specify silhouette_score as int_measure and 'max' as select parameter.
# It returns silhouette score (sil_score), number of clusters selected (sil_best), and silhouette labels (sil_labels)
logging.info("Silhouette score based selection")
sil_score, sil_best, sil_label = select_best(data, modalities, covariates, c, silhouette_score,
                                                      select='max', nclust_range=list(range(2,6)))

logging.info(f"Best number of clusters (and scores): "
              f"{{{sil_best}({sil_score})}}")
# logging.info(f'AMI (true labels vs clustering labels) training = '
#               f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(sil_label)))}')
logging.info('\n\n')

# Save results obtained with silhouette score.
# Silhouette score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
sil_results = {'sil_score': sil_score,
                'sil_best': sil_best,
                'sil_label': sil_label}
pkl.dump(sil_results, open('./sil_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./sil_results_model_name.pkl')


# DAVIES-BOULDIN SCORE
# Use select_best to calcluate Davies-Bouldin score. Specify davies_bouldin_score as int_measure and 'min' as select parameter.
# It returns davies-bouldin score (db_score), number of clusters selected (db_best), and davies-bouldin labels (db_labels)
logging.info("Davies-Bouldin score based selection")
db_score, db_best, db_label = select_best(data, modalities, covariates, c, davies_bouldin_score,
                                                      select='min', nclust_range=None)

logging.info(f"Best number of clusters (and scores): "
              f"{{{db_best}({db_score})}}")
# logging.info(f'AMI (true labels vs clustering labels) training = '
#               f'{adjusted_mutual_info_score(tr_lab, kuhn_munkres_algorithm(np.int32(tr_lab), np.int32(db_label)))}')
logging.info('\n\n')

# Save results obtained with Davies-Bouldin score.
# Davies-Bouldin score, number of clusters, and clusters' labels are organized as a dictionary and saved as a pickle object
db_results = {'db_score': db_score,
                'db_best': db_best,
                'db_label': db_label}
pkl.dump(db_results, open('./db_results_model_name.pkl', 'wb'))
# To open the results, uncomment the following line
# pd.read_pickle('./db_results_model_name.pkl')


### EXTERNAL VALIDATION

# Import data and covariates files for the test set
database = 'database_name.xlsx'
covariates_file = 'covariates_name.xlsx'
# If there are multiple sheets, specify the name of the current sheet 
data = pd.read_excel(os.path.join(data_path,database))
cov = pd.read_excel(os.path.join(data_path, covariates_file))
# Specify the number of subjects belonging to the training set
training_dim = 60
tr_idx = list(range(0,training_dim))
val_idx = list(range(training_dim-1,len(data)-1))

print(f"Training set sample size: {len(tr_idx)}")
print(f"Validation set sample size: {len(val_idx)}")

# Define two dictionaries for modalities and covariates variables.
# For each kind of modality, specify the indeces of the columuns related to the features to be used for clustering
# You can also specify different covaiates for each kind of modality
modalities = {'modality_name_01': data.iloc[:,2:124],
              'modality_name_02': data.iloc[:,124:272],
              'modality_name_03': data.iloc[:,272:]}

covariates = {'modality_name_01': cov.iloc[:,2:],
              'modality_name_02': cov.iloc[:,2:-2],
              'modality_name_03': cov.iloc[:,2:-1]}

# Validate the identified clusters
# Parameters to be specified:
# data: dataset
# modalities: dictionary specifying the features for each modality
# covariates: dictionary specifying the covariates for each modality
# tr_idx: number of training samples
# val_idx: number of validation samples
# nclust: specify the best number of clusters idenfied in the training dataset (i.e., bestncl)
out, mv_tr_embedding, mv_ts_embedding = findbestclust.evaluate_multimodal(data, modalities, covariates, tr_idx, val_idx, nclust=bestncl)
print(f"Training ACC: {out.train_acc}, Test ACC: {out.test_acc}")

# Save cluster labels in the validation set
labels_val = pd.DataFrame(out.test_cllab, columns=['Validation_labels'])
labels_val.to_csv('Labels_val_model_name.csv', index=True)

# Save embeddings for training and validation sets
df_mv_tr_embedding = pd.DataFrame(mv_tr_embedding, index=None)
df_mv_ts_embedding = pd.DataFrame(mv_ts_embedding, index=None)
df_mv_tr_embedding.to_csv('Embedding_training_model_name.csv', index=None)
df_mv_ts_embedding.to_csv('Embedding_validation_model_name.csv', index=None)


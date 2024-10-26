# Multiview_NeuReval
Extension of stability-based relative clustering validation method implementing multiview clustering to determine the best number of clusters based on multimodal data.

## Table of contents
1. [Project Overview](#Project_Overview)
2. [Installation and Requirements](#Installation)
3. [How to use Multiview_NeuReval](#Use)
    1. [Input structure](#Input)
    2. [Grid-search cross-validation for parameters' tuning](#Grid-search)
    3. [Run Multiview_NeuReval with opitmized clustering/classifier/preprocessing algorithms](#NeuReval)
    4. [Compute internal measures](#Internal_measures)
4. [Example](#Example)
5. [Notes](#Notes)
6. [References](#References)

## 1. Project overview <a name="Project_Overview"></a>
*Multiview_NeuReval* extends the stability-based clustering validation approach implemented in ** (https://github.com/fede-colombo/NeuReval/) by implementing multiview clustering algortihm to find the best clustering solution that best generalize to unseen obsrevations in case of multimodal data. Multiview data, in which each sample is represented by multiple views of distinct features, are often seen in real-world data, and related methods have grown in popularity. Multi-view clustering is a flourishing field in unsupervised learning that considers leveraging multiple views of data objects in order to arrive at a more effective and accurate grouping than what can be achieved by just using one view of data.

This package allows to:
1. Select any classification algorithm from *sklearn* library;
2. Select a multiview clustering algorithm from *mvlearn* library (i.e., Multiview KMeans, Multiview Spectral Clustering);
3. Perform (repeated) k-fold cross-validation to determine the best number of clusters;
4. Standardization and covariates adjustement within cross-validation;
5. Combine different kind of multimodal data and apply different set of covariates to each modality;

## 2. Installation and Requirements <a name="Installation"></a>
*Multiview_NeuReval* can be installed with pip:

```python
pip install mvneureval
```
Dependencies useful to install *Multiview_NeuReval* can be found in requirements.txt

## 3. How to use Multiview_NeuReval <a name="Use"></a>
### i. Input structure <a name="Input"></a>
*Multiview_NeuReval* requires that input features and covariates are organized as file excel in the following way:

for database with **input features (database.xlsx)**:
- First column: subject ID
- Second column: diagnosis (e.g., patients=1, healthy controls=0). In case *Multiview_NeuReval* is run on a single diagnostic group, provide a costant value for all subjects.
- From the third column: features

Example of database structure for input features:

| Subject_ID  | Diagnosis | Feature_01 | Feature_02 |
| ------------| ----------| -----------| -----------|
| sub_0001    | 0         | 0.26649221 | 2.13888054 |
| sub_0002    | 1         | 0.32667590 | 0.67116539 |
| sub_0003    | 0         | 0.35406757 | 2.35572978 |

for database with **covariates (covariates.xlsx)**:
- First column: subject ID
- Second column: diagnosis (e.g., patients=1, healhty controls=0). In case *Multiview_NeuReval* is run on a single diagnostic group, provide a costant value for all subjects.
- From the third column: covariates

Example of database structure for covariates:

| Subject_ID  | Diagnosis | Age | Sex | TIV     |
| ------------| ----------| ----|-----| --------|
| sub_0001    | 0         | 54  | 0   | 1213.76 |
| sub_0002    | 1         | 37  | 1   | 1372.93 |
| sub_0003    | 0         | 43  | 0   | 1285.88 |

Templates for both datasets are provided in the folder **Multiview_NeuReval/example_data**.

### ii. Grid-search cross-validation for parameters' tuning <a name="Grid-search"></a>
First, parameters for fixed classifierand multiview clustering algorithms can be optimized through a grid-search cross-validation. This can be done with the ```ParamSelectionMultiview``` class:
```python
ParamSelectionMultiview(params, cv=2, s=s, c=c, nrand=10, n_jobs=-1, iter_cv=1, clust_range=None, strat=None)
```
Parameters to be specified:
- **params**: dictionary of dictionaries of the form {‘s’: {classifier parameter grid}, ‘c’: {clustering parameter grid}} including the lists of classifiers and multiview clustering methods to fit to the data. 
- **cv**: cross-validation folds
- **s**: classifier object
- **c**: multiview clustering object
- **nrand**: number of random labelling iterations, default 10
- **n_jobs**: number of jobs to run in parallel, default (number of cpus - 1)
- **iter_cv**: number of repeated cross-validation, default 1
- **clust_range**: list with number of clusters, default None
- **strat**: stratification vector for cross-validation splits, default ```None```

Once the ```ParamSelectionMultiview``` class is initialized, the ```fit(data,modalities,covariates)``` class method can be used to run grid-search cross-validation.
It returns the optimal number of clusters (i.e., minimum normalized stability), the corresponding normalized stability, and the selected classifier and multiview clustering parameters.
Parameters to be specified:
- **data**: input dataframe
- **modalities**: pyhton dictionary where, for each type of input data, specify the indexes of the columuns related to the features to be used for clustering
- **covariates**: pyhton dictionary where, for each type of input data, specify the indexes of the columuns related to the covariates for each type of input features spcified in **modalities**

### iii. Run Multiview_NeuReval with opitmized multiview clustering and classifier algorithms <a name="NeuReval"></a>
After the selection of the best parameters through grid-search cross-vallidation, we can initalize the ```FindBestClustCVMultiview``` class to assess the normalized stability associated to the best clustering solution and the corresponding clusters' labels

```python
FindBestClustCVCMultiview(s, c, nrand=10, nfold=2, n_jobs=-1, nclust_range=None)
```
Parameters to be specified:
- **s**: classifier object (with opitmized parameters)
- **c**: clustering object (with optimized parameters)
- **nrand**: number of random labelling iterations, default 10
- **nfold**: number of cross-validation folds, default 2
- **n_jobs**: number of jobs to run in parallel, default (number of cpus - 1)
- **clust_range**: list with number of clusters, default None

Once the class has been initialized, the ```best_nclust(data, modalities, covariates, iter_cv=1, strat_vect=None)``` method can be used to obtain the normalized stability, the number of clusters associated to the optimal clustering solution, and clusters' labels. It returns:
- **metrics**: normalized stability
- **bestncl**: best number of clusters
- **tr_lab**: clusters' labels

### iv. Compute internal measures <a name="Internal_measures"></a>
Together with normalized stability, *Multiview_NeuReval* also allows to compute internal measures for comparisons between the stability-based relative validation and internal validation approaches. This can be done with the ```mvneureval.internal_baselines_multiview``` method and the function ```select_best``` to select the best number of clusters that maximize/minimize the selected internal measure:

```python
mvneureval.internal_baselines_confounds.select_best(ata, modalities, covariates, c, silhouette_score,
                                                      select='max', nclust_range=None)
 ```
 Parameters to be specified:
 - **data**: features dataset
 - **modalities**: pyhton dictionary where, for each type of input data, specify the indexes of the columuns related to the features to be used for clustering
 - **covariates**: pyhton dictionary where, for each type of input data, specify the indexes of the columuns related to the covariates for each type of input features spcified in **modalities**
 - **c**: clustering algorithm class (with optimized parameters)
 - **int_measure**: internal measure function (e.g., silhouette score, Davies-Bouldin score)
 - **select**: it can be ‘min’, if the internal measure is to be minimized or ‘max’ if the internal measure should be maximized
 - **nclust_range**: range of clusters to consider, default ```None```


## 4. Example <a name="Example"></a>
An example of how to perform *Multiview_NeuReval* can be found in the folder **Multiview_NeuReval/scripts**. These codes show the application of *Multiview_NeuReval* using Multiview Spectral Clustering as clustering algorithm and Support Vector Machine as classifier:

- **01_grid_search_multiview**: code to perform grid-search cross-validation for parameters tuning
- **02_run_findbestclustcv_multiview**: code to perform *Multiview_NeuReval* with the optimized  algorithms. This script also provides codes to compute different kind of internal measures

## 5. Notes <a name="Notes"></a>
*Multiview_NeuReval* was developed in Python 3.8.10 and tested on ubuntu 20.04. In case of any issues in running *Multiview_NeuReval* on other operating systems (i.e., Windows), you can send an email at the following address: fcolombo.italia@gmail.com

## 6. References <a name="References"></a>
- Landi, I., Mandelli, V., & Lombardo, M. V. (2021). reval: A Python package to determine best clustering solutions with stability-based relative clustering validation. _Patterns_, 2(4), 100228.
- Perry, Ronan, et al. "mvlearn: Multiview Machine Learning in Python." Journal of Machine Learning Research 22.109 (2021): 1-7.

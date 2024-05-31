import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from typing import Tuple

sys.path.append('../src/')
from configs import PARAMS_GB, SEED, NUM_FOLDS

from scipy.stats import entropy
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from sklearn.model_selection import KFold
from scipy.stats import pearsonr, spearmanr
from sklearn.feature_selection import mutual_info_regression

from scipy.cluster import hierarchy
from collections import defaultdict
from scipy.spatial.distance import squareform
import matplotlib.pyplot as plt

def compute_redundancy(X, method='pearson', n_jobs=-1, verbose=0):
    valid_methods = ['pearson', 'spearman', 'mutual_info', 'model_based']
    if method not in valid_methods:
        raise ValueError(f'Invalid method: {method}. Expected one of {valid_methods}.')
    
    is_dataframe = isinstance(X, pd.DataFrame)
    if not is_dataframe:
        X = pd.DataFrame(X)
    features = X.columns.tolist()
    
    if method in ['pearson', 'spearman']:
        corr_matrix = X.corr(method=method)
    elif method == 'mutual_info':
        corr_matrix = pd.DataFrame(data=create_nmi_matrix(X.values, n_jobs=n_jobs, verbose=verbose),
                                   index=features, columns=features)
    elif method == 'model_based':
        corr_matrix = pd.DataFrame(data=create_model_based_matrix(X.values, n_jobs=n_jobs, verbose=verbose),
                                   index=features, columns=features)
    
    return corr_matrix

def calculate_normalized_mi(X, Y):
    """Calculate normalized mutual information."""
    mi = mutual_info_regression(X, Y, n_neighbors=5)
    h_x = entropy(np.histogram(X, bins='auto')[0])
    h_y = entropy(np.histogram(Y, bins='auto')[0])
    return np.clip(mi / np.sqrt(h_x * h_y), 0, 1)

def nmi_pair(data, i, j):
    """Calculate the NMI for a pair of features."""
    return (i, j, calculate_normalized_mi(data[:, i][:,np.newaxis], data[:, j]))

def create_nmi_matrix(data, n_jobs=-1, verbose=0):
    """Create a normalized mutual information matrix for the dataset."""
    n_features = data.shape[1]
    nmi_matrix = np.zeros((n_features, n_features))
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(nmi_pair)(data, i, j) 
                                                      for i in range(n_features) 
                                                      for j in range(i + 1, n_features))

    # Filling the symmetric NMI matrix
    for i, j, nmi_value in results:
        nmi_matrix[i, j] = nmi_matrix[j, i] = nmi_value

    # Setting the diagonal to 1
    np.fill_diagonal(nmi_matrix, 1)

    return nmi_matrix

def r2_score_lgb(preds, train_data):
    metric_name = 'r2'
    y_true = train_data.get_label()
    y_pred = preds
    value = r2_score(y_true, y_pred)
    is_higher_better = True
    return metric_name, value, is_higher_better

def calculate_model_based_dependance(X, Y, num_folds, seed, params):
    
    kf = KFold(n_splits=num_folds, shuffle=True, random_state=seed)
    custom_cv = []
    for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(kf.split(X[:,np.newaxis], Y)):
        custom_cv.append((train_fold_idx, valid_fold_idx))
    
    train_lgb = lgb.Dataset(X[:,np.newaxis], Y)

    cv_results = lgb.cv(params={**params, **{'objective': 'regression', 'num_threads':1, 'metric': None}},
                        train_set=train_lgb,
                        folds=custom_cv,
                        #metrics=None,
                        num_boost_round=params['num_iterations'],
                        #stratified=False,
                        #callbacks=callbacks,
                        eval_train_metric=False,
                        return_cvbooster=False,
                        feval=r2_score_lgb
                       )

    redundancy_feat1_feat2 = np.maximum(cv_results['valid r2-mean'][-1], 0.0)

    train_lgb = lgb.Dataset(Y[:,np.newaxis], X) 

    cv_results = lgb.cv(params={**params, **{'objective': 'regression', 'num_threads':1, 'metric': None}},
                        train_set=train_lgb,
                        folds=custom_cv,
                        #metrics=None,
                        num_boost_round=params['num_iterations'],
                        #stratified=False,
                        #callbacks=callbacks,
                        eval_train_metric=False,
                        return_cvbooster=False,
                        feval=r2_score_lgb
                       )
    redundancy_feat2_feat1 = np.maximum(cv_results['valid r2-mean'][-1], 0.0)

    return (redundancy_feat1_feat2+redundancy_feat2_feat1)/2

def model_based_dependance_pair(data, i, j, num_folds, seed, params):
    """Calculate the model based dependance for a pair of features."""
    return (i, j, calculate_model_based_dependance(data[:, i], data[:, j], num_folds, seed, params))

def create_model_based_matrix(data, n_jobs=-1, verbose=0):
    """Create a model based dependance matrix for the dataset."""
    n_features = data.shape[1]
    dependance_matrix = np.zeros((n_features, n_features))
    
    # Parallel computation
    results = Parallel(n_jobs=n_jobs, verbose=verbose)(delayed(model_based_dependance_pair)
                                                       (data, i, j, num_folds=NUM_FOLDS, seed=SEED, params=PARAMS_GB)
                                                       for i in range(n_features)
                                                       for j in range(i + 1, n_features))
    
    # Filling the symmetric model based matrix
    for i, j, dependance_value in results:
        dependance_matrix[i, j] = dependance_matrix[j, i] = dependance_value

    # Setting the diagonal to 1
    np.fill_diagonal(dependance_matrix, 1)

    return dependance_matrix

def features_clustering(corr_matrix, show=True):
    is_dataframe = isinstance(corr_matrix, pd.DataFrame)
    features = corr_matrix.columns.tolist() if is_dataframe else list(range(corr_matrix.shape[1]))
    corr_matrix = corr_matrix.values if is_dataframe else corr_matrix
    distance_matrix = 1 - np.abs(corr_matrix)
    dist_linkage = hierarchy.ward(squareform(distance_matrix))
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16,8))
        dendro = hierarchy.dendrogram(dist_linkage, labels=features, ax=ax1, leaf_rotation=90)
        ax1.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=True, top=False, labelbottom=True)
        dendro_idx = np.arange(0, len(dendro["ivl"]))
        ax2.imshow(corr_matrix[dendro["leaves"], :][:, dendro["leaves"]])
        ax2.tick_params(axis='both', which='both', bottom=False, left=False, labelleft=True, top=False, labelbottom=True)
        fig.tight_layout()
        plt.show()
    return dist_linkage
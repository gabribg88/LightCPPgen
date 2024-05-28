import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import r2_score
from joblib import Parallel, delayed
from training import *
from sklearn.model_selection import KFold

def find_irrelevant_MDI(results, features):
    importance_dict_df = plot_importance(results, features, show=False, return_imps=True)
    importance_df = importance_dict_df['split']['train'].copy()
    to_drop = importance_df[importance_df['mean'] == 0.0].index.tolist()
    return to_drop

def r2_score_lgb(preds, train_data):
    metric_name = 'r2'
    y_true = train_data.get_label()
    y_pred = preds
    value = r2_score(y_true, y_pred)
    is_higher_better = True
    return metric_name, value, is_higher_better

def evaluate_candidate(train,
                       test,
                       selected_features,
                       candidate,
                       target,
                       num_folds,
                       num_repeats,
                       params,
                       seed,
                       threshold,
                       refit_multiplier,
                       train_score,
                       scoring):
    features = selected_features + [candidate]
    results = cross_validate(train=train,
                             test=test,
                             features=features,
                             target=target,
                             num_folds=num_folds,
                             num_repeats=num_repeats,
                             params=params,
                             threshold=threshold,
                             seed=35,
                             refit=True,
                             refit_multiplier=refit_multiplier,
                             train_score=train_score,
                             train_importances=False,
                             valid_importances=False,
                             test_importances=False,
                             verbose=False)
    score_df = print_results(results, display_metrics=False, return_metrics=True)
    train_score, valid_score, test_enseble_score, test_refit_score = score_df.loc[['Train', 'Valid', 'Test_ensemble', 'Test_refit'], scoring].values
    return candidate, train_score, valid_score, test_enseble_score, test_refit_score

def forward_feature_selection_parallel(train,
                                       test,
                                       features,
                                       target,
                                       num_folds,
                                       num_repeats,
                                       seed,
                                       params,
                                       threshold,
                                       refit_multiplier=1.0,
                                       train_score=True,
                                       min_features_to_select=1,
                                       max_features_to_select=None,
                                       scoring='accuracy',
                                       n_jobs=-1,
                                       verbose=0):
    if max_features_to_select is None:
        max_features_to_select = len(features)
    
    selected_features = []
    remaining_features = features.copy()
    current_score = 0.0
    res = pd.DataFrame(columns=['selected_features', 'train_score', 'valid_score', 'test_ensemble_score', 'test_refit_score'])
    
    while remaining_features and len(selected_features) < max_features_to_select:
        scores_with_candidates = Parallel(n_jobs=n_jobs, verbose=verbose)(
            delayed(evaluate_candidate)(train=train,
                                        test=test,
                                        selected_features=selected_features,
                                        candidate=candidate,
                                        target=target,
                                        num_folds=num_folds,
                                        num_repeats=num_repeats,
                                        seed=seed,
                                        params=params,
                                        threshold=threshold,
                                        refit_multiplier=refit_multiplier,
                                        train_score=train_score,
                                        scoring=scoring)
            for candidate in remaining_features)
        
        #scores_with_candidates.sort()
        #best_score_oof, best_score_test_ensemble, best_score_test_refit, best_candidate = scores_with_candidates[-1]
        scores_df = pd.DataFrame(scores_with_candidates, columns=['feature', 'train_score', 'valid_score', 'test_ensemble_score', 'test_refit_score']).set_index('feature')
        scores_df['score'] = scores_df['valid_score'].values
        
        best_score, best_candidate = scores_df.agg(['max', 'idxmax'])['score'].values
        if best_score > current_score or len(selected_features) < min_features_to_select:
            remaining_features.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_score
        elif len(selected_features) >= min_features_to_select:
            break

        res.loc[len(selected_features), 'selected_features'] = selected_features.copy()
        res.loc[len(selected_features), 'train_score'] = scores_df.loc[best_candidate, 'train_score']
        res.loc[len(selected_features), 'valid_score'] = scores_df.loc[best_candidate, 'valid_score']
        res.loc[len(selected_features), 'test_ensemble_score'] = scores_df.loc[best_candidate, 'test_ensemble_score']
        res.loc[len(selected_features), 'test_refit_score'] = scores_df.loc[best_candidate, 'test_refit_score']
        
        if verbose == 1:
            sys.stderr.write("\rFeatures: %d/%s" % (len(selected_features), max_features_to_select))
            sys.stderr.flush()
        elif verbose > 1:
            sys.stderr.write("\n[%s] Features: %d/%s -- selected: %s -- score: %.4f" % (
                datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                len(selected_features),
                max_features_to_select,
                best_candidate,
                current_score))
            sys.stderr.flush()
    return res, selected_features
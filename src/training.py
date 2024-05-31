import sys
import itertools
import os, random
import numpy as np
import pandas as pd
import lightgbm as lgb
from functools import reduce
import matplotlib.pyplot as plt
from IPython.display import display
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, KFold, LeaveOneOut
from sklearn.metrics import roc_auc_score, matthews_corrcoef, accuracy_score, precision_score, recall_score, f1_score, fbeta_score

def seed_everything(seed=0):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    
def load_dataset(feature_list, dataset_name, features_folder):
    key = ['ID', 'Sequence', 'CPP', 'Dataset']
    dfList = []
    features_dict = {}

    for feature in feature_list:
        tmp = pd.read_pickle(os.path.join(features_folder, f'{dataset_name}_' + feature.split(' ')[0] + '.pickle'))
        features_dict[feature] = list(set(tmp.columns.tolist()) - set(key))
        dfList.append(tmp)

    df = reduce(lambda df1, df2: pd.merge(df1, df2, on=['ID', 'Sequence', 'CPP', 'Dataset']), dfList)

    feature_names = sorted(list(itertools.chain.from_iterable(list(features_dict.values()))))
    return df, feature_names

def create_folds(train, features, target, num_folds, num_repeats=None, shuffle=True, seed=42):
    folds = []
    if num_repeats is None:
        if num_folds > 0:
            skf = StratifiedKFold(n_splits=num_folds, shuffle=shuffle, random_state=seed)
            for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(skf.split(train[features], train[target])):
                folds.append((train_fold_idx, valid_fold_idx))
        elif num_folds == -1:
            loo = LeaveOneOut()
            for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(loo.split(train[features], train[target])):
                folds.append((train_fold_idx, valid_fold_idx))
    else:
        rskf = RepeatedStratifiedKFold(n_splits=num_folds, n_repeats=num_repeats, random_state=seed)
        for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(rskf.split(train[features], train[target])):
            folds.append((train_fold_idx, valid_fold_idx))
    return folds

def cross_validate(train,
                   test,
                   features,
                   target,
                   num_folds,
                   num_repeats,
                   seed,
                   params,
                   threshold=0.5,
                   feval=None,
                   refit=True,
                   refit_multiplier=1.0,
                   train_score=True,
                   train_importances=True,
                   valid_importances=True,
                   test_importances=True,
                   importance_type_list=['split'],
                   custom_cv=None,
                   verbose=True,
                   log=100):
    train = train.copy()
    test = test.copy()
    assert(np.array_equal(train.index.values, np.arange(train.shape[0])))

    if verbose is False:
        log=0

    if custom_cv is None:
        custom_cv = create_folds(train=train, features=features, target=target, num_folds=num_folds, num_repeats=num_repeats, shuffle=True, seed=seed)

    train_lgb = lgb.Dataset(train[features], train[target], feature_name=features, free_raw_data=False, categorical_feature=[])
    test_lgb = lgb.Dataset(test[features], test[target], reference=train_lgb)    

    callbacks = [lgb.log_evaluation(period=log, show_stdv=True),
                 lgb.early_stopping(stopping_rounds=params['early_stopping_round'], first_metric_only=False, verbose=verbose)]

    cv_results = lgb.cv(params=params,
                        train_set=train_lgb,
                        folds=custom_cv,
                        metrics=params['metric'],
                        num_boost_round=params['num_iterations'],
                        #stratified=False,
                        callbacks=callbacks,
                        eval_train_metric=True,
                        return_cvbooster=True,
                        feval=feval
                       )
    best_iteration = cv_results['cvbooster'].best_iteration
    
    results = {'score': {}, 'best_iteration': best_iteration, 'models': {}, 'preds': {}, 'feature_importance': {}}
    results['models']['cv_models'] = cv_results['cvbooster']
    results['score']['train'] = []
    results['score']['valid'] = []
    results['score']['test'] = []
    
    if train_importances:
        results['feature_importance']['train'] = {}
        for importance_type in importance_type_list:
            results['feature_importance']['train'][importance_type] = []
    if valid_importances:
        results['feature_importance']['valid'] = {}
        for importance_type in importance_type_list:
            results['feature_importance']['valid'][importance_type] = []
    if test_importances:
        results['feature_importance']['test'] = {}
        for importance_type in importance_type_list:
            results['feature_importance']['test'][importance_type] = []
            
    train['preds'] = 0.
    train['preds_proba'] = 0.
    test['preds_ensemble'] = 0.
    test['preds_proba_ensemble'] = 0.

    for n_fold, (train_fold_idx, valid_fold_idx) in enumerate(custom_cv):
        train.loc[valid_fold_idx, 'fold'] = n_fold+1
        train_fold = train.loc[train_fold_idx].copy()
        valid_fold = train.loc[valid_fold_idx].copy()
        model = cv_results['cvbooster'].boosters[n_fold]
        ## Train prediction
        if train_score:
            train_preds_proba = pd.Series(model.predict(train_fold[features], num_iteration=best_iteration), index=train_fold.index)
            train_preds = train_preds_proba.apply(lambda x: x>=threshold).astype(int)
            train_score_fold = compute_metrics(true=train_fold[target].values,
                                               preds=train_preds.values, 
                                               preds_proba=train_preds_proba.values, 
                                               fold=f'fold{n_fold+1}')
            results['score']['train'].append(train_score_fold)
        ## Valid prediction
        train.loc[valid_fold_idx, 'preds_proba'] = model.predict(valid_fold[features], num_iteration=best_iteration)
        train.loc[valid_fold_idx, 'preds'] = train.loc[valid_fold_idx, 'preds_proba'].apply(lambda x: x>=threshold).astype(int)
        valid_score_fold = compute_metrics(true=valid_fold[target].values,
                                           preds=train.loc[valid_fold_idx, 'preds'].values, 
                                           preds_proba=train.loc[valid_fold_idx, 'preds_proba'].values, 
                                           fold=f'fold{n_fold+1}')
        results['score']['valid'].append(valid_score_fold)
        ## Test prediction
        test['preds_proba_ensemble'] += model.predict(test[features], num_iteration=best_iteration) / num_folds
        
        ## Feature importance
        if train_importances:
            for importance_type in importance_type_list:
                if importance_type in ['split', 'gain']:
                    results['feature_importance']['train'][importance_type].append(model.feature_importance(importance_type=importance_type, iteration=best_iteration))
                elif importance_type == 'shap':
                    results['feature_importance']['train']['shap'].append(model.predict(train_fold[features], num_iteration=best_iteration, pred_contrib=True))
        if valid_importances:
            if importance_type == 'shap':
                results['feature_importance']['valid']['shap'].append(model.predict(valid_fold[features], num_iteration=best_iteration, pred_contrib=True))
        if test_importances:
            if importance_type == 'shap':
                results['feature_importance']['test']['shap'].append(model.predict(test[features], num_iteration=best_iteration, pred_contrib=True))

    results['preds']['train'] = train
    test['preds_ensemble'] = test['preds_proba_ensemble'].apply(lambda x: x>=threshold).astype(int)
    results['score']['test'].append(compute_metrics(true=test[target].values,
                                                    preds=test['preds_ensemble'].values, 
                                                    preds_proba=test['preds_proba_ensemble'].values, 
                                                    fold='Test_ensemble'))

    if refit:
        train_lgb = lgb.Dataset(train[features], train[target], feature_name=features, free_raw_data=False, categorical_feature=[])
        test_lgb = lgb.Dataset(test[features], test[target], reference=train_lgb)    

        history = dict()
        callbacks = [lgb.log_evaluation(period=log, show_stdv=True),
                     lgb.record_evaluation(history)]

        if params['boosting_type'] == 'dart':
            model = lgb.train(params=params,
                              train_set=train_lgb,
                              valid_sets = [train_lgb, test_lgb],
                              callbacks=callbacks,
                              #num_boost_round=best_iteration,
                              feval=feval)
        else:
            model = lgb.train(params=dict(params, **{'num_iterations': int(best_iteration*refit_multiplier), 'early_stopping_round': None}),
                              train_set=train_lgb,
                              valid_sets = [train_lgb, test_lgb],
                              callbacks=callbacks,
                              num_boost_round=best_iteration,
                              feval=feval)

        test['preds_proba_refit'] = model.predict(test[features], num_iteration=int(best_iteration*refit_multiplier))
        test['preds_refit'] = test['preds_proba_refit'].apply(lambda x: x>=threshold).astype(int)
        results['score']['test'].append(compute_metrics(true=test[target].values,
                                                        preds=test['preds_refit'].values, 
                                                        preds_proba=test['preds_proba_refit'].values, 
                                                        fold='Test_refit'))

        results['models']['refit_model'] = model

    results['preds']['test'] = test
    return results

def print_results(results,
                  display_metrics=True,
                  return_metrics=False): 
    score_df_list = []
    for dataset, score in results['score'].items():
        score_df = pd.concat(score)
        if dataset in ['train', 'valid']:
            score_df.loc[f'{dataset.capitalize()}'] = score_df.mean(axis=0)
        score_df_list.append(score_df)

    score_df_concat = pd.concat(score_df_list)
    if display_metrics:
        try:
            display(score_df_concat.loc[['Train', 'Valid', 'Test_ensemble', 'Test_refit']])
        except:
            display(score_df_concat.loc[['Train', 'Valid', 'Test_ensemble']])
    if return_metrics:
        return score_df_concat

def plot_importance(results,
                    features=None,
                    max_features=None,
                    show=True,
                    return_imps=False):
    importance_dict_df = {}
    for importance_type in ['split', 'gain', 'shap', 'permutation']:
        importance_dict_df[importance_type] = {}

    for dataset, importance_dict in results['feature_importance'].items():
        for importance_type, importances in importance_dict.items():
            if len(importances) > 0:
                if importance_type in ['split', 'gain']:
                    importance_df = pd.concat([pd.DataFrame(importances[i], index=features, columns=[f'Fold{i+1}']) for i in range(len(importances))], axis=1)
                    importance_df = pd.DataFrame({'mean': importance_df.mean(axis=1), 'std': importance_df.std(axis=1)}).sort_values(by='mean', ascending=False) # faster then agg
                elif importance_type in ['shap']:
                    importance_df = [pd.DataFrame(importances[i], columns=features + ['expected_values'])\
                                [features].abs().mean(axis=0).to_frame(name=f'Fold{i+1}') for i in range(len(importances))]
                    importance_df = reduce(lambda df1,df2: pd.merge(df1,df2,left_index=True, right_index=True), importance_df)
                    importance_df = pd.DataFrame({'mean': importance_df.mean(axis=1), 'std': importance_df.std(axis=1)}).sort_values(by='mean', ascending=False) # faster then agg
                if max_features is not None:
                    importance_df = importance_df.iloc[:max_features]
                importance_dict_df[importance_type][dataset] = importance_df

    if show:
        if len(importance_dict_df['shap']) > 0:
            n_cols = len(importance_dict_df['shap'])
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5), sharex=True)
            for i, (dataset, importance_df) in enumerate(importance_dict_df['shap'].items()):
                axes[i].barh(y=importance_df.index.values, width=importance_df['mean'].values, xerr=importance_df['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
                axes[i].set_title(f'{dataset.capitalize()} shap feature importance')
                axes[i].invert_yaxis()
                axes[i].set_axisbelow(True)
            plt.tight_layout()
            plt.show()
        if len(importance_dict_df['permutation']) > 0:
            n_cols = len(importance_dict_df['permutation'])
            fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5), sharex=True)
            for i, (dataset, importance_df) in enumerate(importance_dict_df['permutation'].items()):
                axes[i].barh(y=importance_df.index.values, width=importance_df['mean'].values, xerr=importance_df['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
                axes[i].set_title(f'{dataset.capitalize()} permutation feature importance')
                axes[i].invert_yaxis()
                axes[i].set_axisbelow(True)
            plt.tight_layout()
            plt.show()
        if len(importance_dict_df['split']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            for i, (dataset, importance_df) in enumerate(importance_dict_df['split'].items()):
                ax.barh(y=importance_df.index.values, width=importance_df['mean'].values, xerr=importance_df['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
                ax.set_title(f'{dataset.capitalize()} split feature importance')
                ax.invert_yaxis()
                ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()
        if len(importance_dict_df['gain']) > 0:
            fig, ax = plt.subplots(1, 1, figsize=(5, 5))
            for i, (dataset, importance_df) in enumerate(importance_dict_df['gain'].items()):
                ax.barh(y=importance_df.index.values, width=importance_df['mean'].values, xerr=importance_df['std'].values, capsize=3, edgecolor='black', linewidth=0.5)
                ax.set_title(f'{dataset.capitalize()} gain feature importance')
                ax.invert_yaxis()
                ax.set_axisbelow(True)
            plt.tight_layout()
            plt.show()

    if return_imps:
        return importance_dict_df

def compute_metrics(true, preds, preds_proba, fold):
    res = pd.DataFrame(data=0.0, index=[fold], columns=['AUROC', 'MCC', 'F1', 'Fb05', 'Fb01', 'ACC', 'SN', 'SP'])
    res['AUROC'] = roc_auc_score(true, preds_proba)
    res['MCC'] = matthews_corrcoef(true, preds)
    res['F1'] = f1_score(true, preds)
    res['Fb05'] = fbeta_score(true, preds, beta=0.5)
    res['Fb01'] = fbeta_score(true, preds, beta=0.1)
    res['ACC'] = accuracy_score(true, preds)
    res['SN'] = recall_score(true, preds)
    res['SP'] = recall_score(true, preds, pos_label=0)
    return res


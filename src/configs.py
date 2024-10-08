####
fig_width = 10
fig_height = 4
fig_size =  [fig_width,fig_height]
fig_params = {'lines.linewidth': 1,
              'axes.grid': True,
              'grid.linestyle': '--',
              'grid.linewidth': 0.5,
              'axes.labelsize': 12,
              'axes.titlesize': 12,
              'font.size': 11,
              'legend.fontsize': 10,
              'xtick.labelsize': 10,
              'ytick.labelsize': 10,
              'figure.figsize': fig_size,
              'savefig.dpi': 600,
              'axes.linewidth': 1.0,
              ##
              'xtick.direction': 'in',
              'xtick.major.size': 3,
              'xtick.major.width': 0.5,
              'xtick.minor.size': 1.5,
              'xtick.minor.width': 0.5,
              'xtick.minor.visible': True,
              'xtick.top': True,
              ##
              'ytick.direction': 'in',
              'ytick.major.size': 3,
              'ytick.major.width': 0.5,
              'ytick.minor.size': 1.5,
              'ytick.minor.width': 0.5,
              'ytick.minor.visible': True,
              'ytick.right': True,
              ##
              'font.family': 'serif',
              'mathtext.fontset': 'dejavuserif',
              }

###################
TARGET = 'CPP'
SEED = 42
NUM_FOLDS = 10
NUM_REPEATS = None


############ LIGHTGBM PARAMETERS ############

###### CORE PARAMETERS
OBJECTIVE = 'binary'
BOOSTING_TYPE = 'gbdt' # default = gbdt
NUM_ITERATIONS = 100000 # default = 100
LEARNING_RATE = 0.1 # default = 0.1
NUM_LEAVES = 31 # default = 31, in XGBoost num_leaves = 2**max_depth
NUM_THREADS = 16#32 # default = 0, 0 means default number of threads in OpenMP, for the best speed, set this to the number of real CPU cores, not the number of threads
DEVICE = 'cpu'

###### LEARNING CONTROL PARAMETERS
FORCE_COL_WISE = 'false' # default = false, set this to true to force col-wise histogram building, enabling this is recommended when the number of columns is large, or the total number of bins is large
FORCE_ROW_WISE = 'false' # default = false, set this to true to force row-wise histogram building enabling this is recommended when the number of data points is large, and the total number of bins is relatively small
MAX_DEPTH = -1 # default = -1, limit the max depth for tree model. This is used to deal with over-fitting when data is small. Tree still grows leaf-wise, <= 0 means no limit
MIN_DATA_IN_LEAF = 20 # default = 20, minimal number of data in one leaf. Can be used to deal with over-fitting
MIN_SUM_HESSIAN_IN_LEAF = 1e-3 #  default = 1e-3, minimal sum hessian in one leaf. Like min_data_in_leaf, it can be used to deal with over-fitting
BAGGING_FRACTION = 1.0 #  default = 1.0, like feature_fraction, but this will randomly select part of data without resampling, can be used to speed up training, can be used to deal with over-fitting
POS_BAGGING_FRACTION = 1.0 # default = 1.0. used for imbalanced binary classification problem, will randomly sample #pos_samples * pos_bagging_fraction positive samples in bagging. set this to 1.0 to disable. should be used together with neg_bagging_fraction
NEG_BAGGING_FRACTION = 1.0 # default = 1.0. used for imbalanced binary classification problem, will randomly sample #neg_samples * neg_bagging_fraction negative samples in bagging. set this to 1.0 to disable. should be used together with pos_bagging_fraction
BAGGING_FREQ = 0 # default = 0, frequency for bagging. 0 means disable bagging; k means perform bagging at every k iteration. Every k-th iteration, LightGBM will randomly select bagging_fraction * 100 % of the data to use for the next k iterations
BAGGING_SEED = SEED # default = 3, random seed for bagging
FEATURE_FRACTION = 1.0 # default = 1.0, LightGBM will randomly select a subset of features on each iteration (tree) if feature_fraction is smaller than 1.0.
                       # For example, if you set it to 0.8, LightGBM will select 80% of features before training each tree, can be used to speed up training, can be used to deal with over-fitting
FEATURE_FRACTION_BYNODE = 1.0 # default = 1.0, LightGBM will randomly select a subset of features on each tree node if feature_fraction_bynode is smaller than 1.0.
                              # For example, if you set it to 0.8, LightGBM will select 80% of features at each tree node, can be used to deal with over-fitting
FEATURE_FRACTION_SEED = SEED # default = 2, random seed for feature_fraction
EARLY_STOPPING = 20 # default = 0, will stop training if one metric of one validation data doesn’t improve in last early_stopping_round rounds, <= 0 means disable
FIRST_METRIC_ONLY = 'false' # default = false, LightGBM allows you to provide multiple evaluation metrics. Set this to true, if you want to use only the first metric for early stopping
LAMBDA_L1 = 0.0 # default = 0.0, type = double, L1 regularization
LAMBDA_L2 = 0.0 # default = 0.0, type = double, L2 regularization
LINEAR_LAMBDA = 0.0 # default = 0.0, type = double, linear tree regularization, corresponds to the parameter lambda in Eq. 3 of Gradient Boosting with Piece-Wise Linear Regression Trees
CAT_L2 = 10.0 # default = 10.0, used for the categorical features, L2 regularization in categorical split
CAT_SMOOTH = 10.0 # default = 10.0, used for the categorical features, this can reduce the effect of noises in categorical features, especially for categories with few data
VERBOSITY = -1 # default = 1, controls the level of LightGBM’s verbosity. < 0: Fatal, = 0: Error (Warning), = 1: Info, > 1: Debug

###### DATASET PARAMETERS
MAX_BIN = 255 # default = 255, max number of bins that feature values will be bucketed in small number of bins may reduce training accuracy but may increase general power (deal with over-fitting)
MIN_DATA_IN_BIN = 3 # default = 3, minimal number of data inside one bin, use this to avoid one-data-one-bin (potential over-fitting)
USE_MISSING = 'true' # default = true, set this to false to disable the special handle of missing value
DATA_RANDOM_SEED = SEED # random seed for sampling data to construct histogram bins

###### OBJECTIVE PARAMETERS
IS_UNBALANCED = 'false' # default = false, used only in binary applications, set this to true if training data are unbalanced, 
                        # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
                        # Note: this parameter cannot be used at the same time with scale_pos_weight, choose only one of them
SCALE_POS_WEIGHT = 1.0 # default = 1.0, used only in binary, weight of labels with positive class
                       # Note: while enabling this should increase the overall performance metric of your model, it will also result in poor estimates of the individual class probabilities
                       # Note: this parameter cannot be used at the same time with is_unbalance, choose only one of them
        
###### METRIC PARAMETERS
METRIC = 'auc'

PARAMS_GB = { 'objective': OBJECTIVE,
              'boosting_type': BOOSTING_TYPE,
              'num_iterations':NUM_ITERATIONS,
              'learning_rate': LEARNING_RATE,
              'num_threads': NUM_THREADS,
              'device': DEVICE,
              'seed': SEED,
              'force_col_wise': FORCE_COL_WISE,
              'force_row_wise':FORCE_ROW_WISE,
              'max_depth': MAX_DEPTH, # max_depth
              'num_leaves': NUM_LEAVES, #  2**max_depth
              'min_data_in_leaf': MIN_DATA_IN_LEAF, # min_data_in_leaf
              'min_sum_hessian_in_leaf': MIN_SUM_HESSIAN_IN_LEAF,
              'bagging_fraction': BAGGING_FRACTION,
              'pos_bagging_fraction': POS_BAGGING_FRACTION,
              'neg_bagging_fraction': NEG_BAGGING_FRACTION,
              'bagging_freq': BAGGING_FREQ,
              'bagging_seed': BAGGING_SEED,
              'feature_fraction': FEATURE_FRACTION,
              'feature_fraction_bynode': FEATURE_FRACTION_BYNODE,
              'feature_fraction_seed': FEATURE_FRACTION_SEED,
              'early_stopping_round': EARLY_STOPPING,
              'first_metric_only': FIRST_METRIC_ONLY,
              'lambda_l1': LAMBDA_L1,
              'lambda_l2': LAMBDA_L2,
              'linear_lambda': LINEAR_LAMBDA,
              'cat_l2': CAT_L2,
              'cat_smooth': CAT_SMOOTH,
              'max_bin': MAX_BIN,
              'min_data_in_bin': MIN_DATA_IN_BIN,
              'use_missing': USE_MISSING,
              'data_random_seed': DATA_RANDOM_SEED,
              'is_unbalance': IS_UNBALANCED,
              'scale_pos_weight': SCALE_POS_WEIGHT,
              'metric': METRIC,
              'verbosity':VERBOSITY
              }

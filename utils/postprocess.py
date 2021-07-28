import pandas as pd
from math import sqrt
import numpy as np
from sklearn.inspection import permutation_importance
from sklearn.base import clone

# collect feature importance values across training folds
def feature_importance_built_in(models, model_name, features):
    print('--- feature_importance_built_in ---')

    d = {'feat': features} # create feature importance dataframe
    col_headers = []

    idx = 1
    for model in models:
        col_header = 'feat_imp_fold' + str(idx)
        if model_name == 'LogisticRegression':
            d[col_header] = model.coef_[0]
        else:
            d[col_header] = model.feature_importances_
        col_headers.append(col_header)
        idx += 1
    
    feat_imp = pd.DataFrame(data=d)
    #print(feat_imp)
    feat_imp['feat_imp_mean'] = feat_imp[col_headers].mean(axis=1)
    feat_imp['feat_imp_std'] = feat_imp[col_headers].std(axis=1)
    feat_imp['feat_imp_zeros'] = (feat_imp[col_headers] == 0).astype(int).sum(axis=1)

    confidence_level = 2.776 # corresponding z-score value for 95% confidence and 4 degrees of freedom (since we do 5 fold cv)
    feat_imp['feat_imp_95_ci'] = confidence_level * feat_imp['feat_imp_std'] / sqrt(len(col_headers))

    feat_imp.sort_values(by='feat_imp_mean', ascending=False, inplace=True)
    print(feat_imp[['feat', 'feat_imp_mean', 'feat_imp_95_ci']])
    return feat_imp

def feature_importance_permutation_importance(model, KFold_data):
    print('--- feature_importance_permutation_importance ---')

    d = {'feat': KFold_data[0]['x_train_fold'].columns} # create feature importance dataframe
    col_headers = []

    for fold in list(KFold_data.keys()):
        col_header = 'feat_imp_fold' + str(fold)
        col_headers.append(col_header)

        fold_data = KFold_data[fold]

        x_train_fold = fold_data['x_train_fold']
        x_test_fold = fold_data['x_test_fold']
        y_train_fold = fold_data['y_train_fold']
        y_test_fold = fold_data['y_test_fold']

        model_copy = clone(model)
        model_copy.fit(x_train_fold, y_train_fold)

        r = permutation_importance(model_copy, x_test_fold, y_test_fold, n_repeats=20)

        d[col_header] = r.importances_mean

    feat_imp = pd.DataFrame(data=d)

    feat_imp['feat_imp_mean'] = feat_imp[col_headers].mean(axis=1)
    feat_imp['feat_imp_std'] = feat_imp[col_headers].std(axis=1)
    feat_imp['feat_imp_zeros'] = (feat_imp[col_headers] == 0).astype(int).sum(axis=1)

    confidence_level = 2.776 # corresponding z-score value for 95% confidence and 4 degrees of freedom (since we do 5 fold cv)
    feat_imp['feat_imp_95_ci'] = confidence_level * feat_imp['feat_imp_std'] / sqrt(len(col_headers))

    feat_imp.sort_values(by='feat_imp_mean', ascending=False, inplace=True)
    print(feat_imp[['feat', 'feat_imp_mean', 'feat_imp_95_ci']])
    return feat_imp

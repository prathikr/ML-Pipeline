import pandas as pd
from math import sqrt
import numpy as np
from sklearn.inspection import permutation_importance

# collect feature importance values across training folds
def feature_importance(models, model_name, strategy, features):
    print('--- feature_importance ---')

    d = {'feat': features} # create feature importance dataframe
    col_headers = []

    idx = 1
    for model in models:
        col_header = 'feat_imp_fold' + str(idx)
        if strategy == 'built_in':
            if model_name == 'LogisticRegression':
                d[col_header] = model.coef_[0]
            else:
                d[col_header] = model.feature_importances_
        col_headers.append(col_header)
        idx += 1
    
    feat_imp = pd.DataFrame(data=d)

    feat_imp['feat_imp_mean'] = feat_imp[col_headers].mean(axis=1)
    feat_imp['feat_imp_std'] = feat_imp[col_headers].std(axis=1)
    feat_imp['feat_imp_zeros'] = (feat_imp[col_headers] == 0).astype(int).sum(axis=1)

    confidence_level = 2.776 # corresponding z-score value for 95% confidence and 4 degrees of freedom (since we do 5 fold cv)
    feat_imp['feat_imp_95_ci'] = confidence_level * feat_imp['feat_imp_std'] / sqrt(len(col_headers))

    feat_imp.sort_values(by='feat_imp_mean', inplace=True)

    return feat_imp
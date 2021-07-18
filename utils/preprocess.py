import pandas as pd

def print_statistics(data_original, outcome):
    print('--- print_statistics ---')
    print('Outcome distribution:', data_original[outcome].value_counts(), sep='\n') # outcome class distribution
    print('NaN counts:', data_original.isna().sum(), sep='\n')

def one_hot_encode(data_original, cols):
    print('--- one_hot_encode ---')
    data = data_original.copy()

    print('Features to encode:', cols)
    for categorical_var in cols:
        # one-hot encode variable
        one_hot = pd.get_dummies(data[categorical_var], prefix=categorical_var)
        data = data.drop(categorical_var, axis = 1)
        data = data.join(one_hot)
    
    print('Data shape after one_hot_encode:', data.shape)
    return data

def impute(data_original, strategy):
    print('--- impute ---')
    data = data_original.copy()

    print('# NaNs pre-imputation:', data.isna().sum().sum())

    for key in strategy:
        cols = strategy[key]
        for col in cols:
            if key == 'mean':
                data[col].fillna((data[col].mean()), inplace=True)
            elif key == 'mode':
                data[col].fillna((data[col].mode()[0]), inplace=True)
            else:
                raise Exception("imputation strategy unsupported")
    print('# NaNs post-imputation:', data.isna().sum().sum())
    if data.isna().sum().sum() > 0:
        print('WARNING: Dataset still contains NaN values post-imputation!!!')
    return data
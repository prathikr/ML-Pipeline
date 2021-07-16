import pandas as pd

def one_hot_encode(data_original, cols):
    print('--- one_hot_encode ---')
    data = data_original.copy()

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

    print("NaN counts pre-imputation:\n", data.isna().sum())

    for col in data.columns:
        data[col].fillna((data[col].mean()), inplace=True)

    print("NaN counts post-imputation:\n", data.isna().sum())
    return data
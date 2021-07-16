import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.modelFactory import ModelFactory

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

class Pipeline():

    def __init__(self, config_filename):
        print("--- __init__ ---")

        with open(config_filename) as f:
            self.config = json.load(f)

        self.modelFactory = ModelFactory({})

    def preprocess(self):
        print("--- preprocess ---")

        self.data = pd.read_csv(self.config['data_filename'], index_col=0)
        print("Original data shape:", self.data.shape)

        if len(self.config['preprocess']['one_hot_encode']) > 0:
            self.data = one_hot_encode(self.data, self.config['preprocess']['one_hot_encode'])

        self.data = impute(self.data, self.config['preprocess']['impute'])

        self.X = self.data.drop(columns=[self.config['outcome']])
        self.y = self.data[self.config['outcome']]

    def model(self):
        print("--- model ---")

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=2017)
        print("X_train.shape", X_train.shape, "X_test.shape", X_test.shape)
        
        model = self.modelFactory.initModel(self.config["model"])
        self.model = model.fit(X_train, y_train)

        print(self.config["model"], "accuracy on test set:", model.score(X_test, y_test))

    def postprocess(self):
        print("--- postprocessing ---")

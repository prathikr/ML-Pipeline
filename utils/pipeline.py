import json
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

def one_hot_encode(data_original):
    print("--- one_hot_encode ---")
    data = data_original.copy()
    cols = ['Sex', 'Embarked']

    for categorical_var in cols:
        if categorical_var in list(data.columns):
            # one-hot encode variable
            one_hot = pd.get_dummies(data[categorical_var], prefix=categorical_var, dummy_na=True)
            data = data.drop(categorical_var, axis = 1)
            data = data.join(one_hot)

    return data

class Pipeline():

    def __init__(self, config_filename):
        print("--- __init__ ---")

        self.config_filename = config_filename
        valid = self.validate_config()
        if not valid:
            raise("pipeline constructor failed")

    def validate_config(self):
        print("--- validate config ---")

        with open(self.config_filename) as f:
            config = json.load(f)
            self.config = config
        print("Config:", self.config)

        data_filename = self.config['data_filename']
        self.data_filename = data_filename

        return True

    def preprocess(self):
        print("--- preprocess ---")

        data = pd.read_csv(self.data_filename, index_col=0)
        self.data = data

        data_one_hot_encoded = one_hot_encode(self.data)
        print(data_one_hot_encoded.columns)

        for col in data_one_hot_encoded.columns:
            data_one_hot_encoded[col].fillna(data_one_hot_encoded[col].mode()[0], inplace=True)

        self.data = data_one_hot_encoded

        self.X = self.data.drop(columns=['Survived'])
        self.y = self.data['Survived']


    def model(self):
        print("--- model ---")

        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=2017)
        print("X_train.shape", X_train.shape, "y_train.shape", y_train.shape)
        
        model = LogisticRegression(max_iter=1000)
        self.model = model.fit(X_train,y_train)

        print(model.score(X_test, y_test))

    def postprocess(self):
        print("--- postprocessing ---")
        return True
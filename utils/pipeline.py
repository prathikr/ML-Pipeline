import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.modelFactory import ModelFactory
from utils.preprocess import *

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

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.20, random_state=2017)
        print("X_train.shape", self.X_train.shape, "X_test.shape", self.X_test.shape)
        
        model = self.modelFactory.initModel(self.config["model"])
        self.model = model.fit(self.X_train, self.y_train)

    def postprocess(self):
        print("--- postprocessing ---")
        print(self.config["model"], "accuracy on test set:", self.model.score(self.X_test, self.y_test))

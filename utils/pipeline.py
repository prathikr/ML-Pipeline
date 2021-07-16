import json
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.modelFactory import ModelFactory
from utils.preprocess import *
from utils.model import *
from utils.postprocess import *

class Pipeline():

    def __init__(self, config_filename):
        with open(config_filename) as f:
            self.config = json.load(f)

    def preprocess(self):
        print("--- preprocess ---")
        self.data = pd.read_csv(self.config['data_filename'], index_col=0)
        print("Original data shape:", self.data.shape)

        preprocess_args = self.config['preprocess']
        self.data = impute(self.data, preprocess_args['imputation_strategy'])

        if len(preprocess_args['one_hot_encode']) > 0:
            self.data = one_hot_encode(self.data, preprocess_args['one_hot_encode'])

        print("Final data shape:", self.data.shape)
        self.X = self.data.drop(columns=[self.config['outcome']])
        self.y = self.data[self.config['outcome']]

    def model(self):
        print("--- model ---")
        self.modelFactory = ModelFactory({})

        model_args = self.config['model']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2)
        print("X_train.shape", self.X_train.shape, "X_test.shape", self.X_test.shape)
        
        model = self.modelFactory.initModel(model_args['name'])

        if model_args['forward_feature_selection']:
            scores, ordered_features = forward_feature_selection(model, self.X_train, self.y_train, model_args['optimization_metric'])
            optimal_n_features = scores.index(max(scores))
            print('\nOptimal number of features:', optimal_n_features)
            optimal_features = ordered_features[:optimal_n_features]

            self.X_train = self.X_train[optimal_features]
            self.X_test = self.X_test[optimal_features]

        self.model = model.fit(self.X_train, self.y_train)

    def postprocess(self):
        print("--- postprocessing ---")

        model_args = self.config['model']
        test_score = evaluate(self.model, self.X_test, self.y_test, model_args['optimization_metric'])
        print(model_args['name'], model_args['optimization_metric'], "on test set:", test_score)

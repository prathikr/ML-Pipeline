import json
import pandas as pd
import numpy as np
from sklearn.model_selection import cross_validate
from utils.modelFactory import ModelFactory
from utils.preprocess import *
from utils.model import *
from utils.postprocess import *
from math import sqrt

class Pipeline():

    def __init__(self, config_filename):
        with open(config_filename) as f:
            self.config = json.load(f)

        self.modelFactory = ModelFactory({})

    def preprocess(self):
        preprocess_args = self.config['preprocess']

        self.data = pd.read_csv(self.config['data_filename'], index_col=0)
        print("Original data shape:", self.data.shape, '\n')

        print_statistics(self.data, self.config['outcome'])
        self.data = impute(self.data, preprocess_args['imputation_strategy'])

        if len(preprocess_args['one_hot_encode']) > 0:
            self.data = one_hot_encode(self.data, preprocess_args['one_hot_encode'])

        print('\nFinal data shape:', self.data.shape)
        self.X = self.data.drop(columns=[self.config['outcome']])
        self.y = self.data[self.config['outcome']]

    def model(self):
        model_args = self.config['model']
        self.model_name = model_args['name']

        model = get_model(self.model_name, model_args['importlib'], model_args['params'])

        if model_args['forward_feature_selection']:
            scores, ordered_features = forward_feature_selection(model, self.X, self.y, model_args['optimization_metric'])
            optimal_n_features = scores.index(max(scores))
            print('Optimal number of features:', optimal_n_features)
            optimal_features = ordered_features[:optimal_n_features]
            print('Optimal features:', optimal_features)

            self.X = self.X[optimal_features]

        self.scores = cross_validate(clone(model), self.X, self.y, cv=5, scoring=[model_args['optimization_metric']], return_estimator=True)
        confidence_level = 2.776 # corresponding z-score value for 95% confidence and 4 degrees of freedom (since we do 5 fold cv)
        print(model_args['optimization_metric'], 'performance across 5 folds with 95% confidence:' ,round(self.scores['test_' + model_args['optimization_metric']].mean(), 4), '+/-', round(confidence_level * self.scores['test_' + model_args['optimization_metric']].std() / sqrt(len(self.scores['test_' + model_args['optimization_metric']])), 4))

    def postprocess(self):
        postprocess_args = self.config['postprocess']

        self.feature_importance = feature_importance(self.scores['estimator'], self.model_name, postprocess_args['feature_importance_strategy'], self.X.columns)
        print(self.feature_importance[['feat', 'feat_imp_mean', 'feat_imp_95_ci']])

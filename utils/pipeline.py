import json
import pandas as pd
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
import numpy as np
#from sklearn.model_selection import cross_validate
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import get_scorer
from utils.preprocess import *
from utils.model import *
from utils.postprocess import *
from math import sqrt

class Pipeline():

    def __init__(self, config_filename):
        with open(config_filename) as f:
            self.config = json.load(f)

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

        #self.scores = cross_validate(clone(model), self.X, self.y, cv=5, scoring=[model_args['optimization_metric']], return_estimator=True)
        # Create StratifiedKFold object.
        skf = StratifiedKFold(n_splits=5, shuffle=True)
        scorer = get_scorer(model_args['optimization_metric'])
        cross_val_scores = []
        self.KFold_data = {}
        self.estimators = []
        fold = 0
        for train_index, test_index in skf.split(self.X, self.y):
            x_train_fold, x_test_fold = self.X.iloc[train_index], self.X.iloc[test_index]
            y_train_fold, y_test_fold = self.y.iloc[train_index], self.y.iloc[test_index]
            self.KFold_data[fold] = {'x_train_fold': x_train_fold, 'x_test_fold': x_test_fold, 'y_train_fold': y_train_fold, 'y_test_fold': y_test_fold}
            model = get_model(self.model_name, model_args['importlib'], model_args['params'])
            self.estimators.append(model.fit(x_train_fold, y_train_fold))
            y_pred = self.estimators[fold].predict(x_test_fold)
            cross_val_scores.append(scorer._score_func(y_test_fold, y_pred))
            fold += 1
        
        confidence_level = 2.776 # corresponding z-score value for 95% confidence and 4 degrees of freedom (since we do 5 fold cv)
        print(model_args['optimization_metric'], 'performance across 5 folds with 95% confidence:', round(np.array(cross_val_scores).mean(), 4), '+/-', round(confidence_level * np.array(cross_val_scores).std() / sqrt(len(cross_val_scores)), 4))

    def postprocess(self):
        postprocess_args = self.config['postprocess']
        model_args = self.config['model']

        if postprocess_args['feature_importance_strategy'] == 'built_in':
            feature_importance_built_in(self.estimators, self.model_name, self.X.columns)
        elif postprocess_args['feature_importance_strategy'] == 'permutation_importance':
            model = get_model(self.model_name, model_args['importlib'], model_args['params'])
            feature_importance_permutation_importance(model, self.KFold_data)

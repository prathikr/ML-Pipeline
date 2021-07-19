from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

class ModelFactory():

    def __init__(self, params):
        self.params = params

    def initModel(self, model_name, params):
        if model_name == 'LogisticRegression':
            return LogisticRegression(**params)
        elif model_name == 'RandomForestClassifier':
            return RandomForestClassifier(**params)
        else:
            raise Exception(f'unsupported model type: {model_name}')
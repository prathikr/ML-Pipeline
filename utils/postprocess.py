from sklearn.base import clone
from sklearn.model_selection import cross_validate
from sklearn.metrics import make_scorer, precision_score

def evaluate(model, X_test, y_test, optimization_metric):
    temp_scores = cross_validate(clone(model), X_test, y_test, cv=5, scoring=[optimization_metric])
    score = temp_scores['test_' + optimization_metric].mean()
    return score
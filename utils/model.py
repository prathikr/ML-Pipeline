from sklearn.metrics import make_scorer, precision_score
from sklearn.model_selection import cross_validate
from sklearn.base import clone
import matplotlib.pyplot as plt

def forward_feature_selection(model, X, y, optimization_metric):
    print('--- forward_feature_selection ---')
    features = list(X.columns)
    selected_features = []
    scores = []
    
    for i in range(X.shape[1]):
        best_score = 0
        next_feat = ''
        for feat in features:
            selected_features.append(feat)
            temp_X = X[selected_features]
            temp_scores = cross_validate(clone(model), temp_X, y, cv=5, scoring=[optimization_metric])
            temp_score = temp_scores['test_' + optimization_metric].mean()
            if temp_score >= best_score:
                best_score = temp_score
                next_feat = feat
            selected_features.pop()
        #print('Added Feature:', next_feat)
        selected_features.append(next_feat)
        features.remove(next_feat)
        scores.append(best_score)
        
    #print('Ordering of Features:', selected_features)
    
    plt.title('Forward Feature Selection')
    plt.xlabel('# of Features')
    plt.ylabel(optimization_metric)
    plt.plot(list(range(X.shape[1])), scores)
    plt.show() # blocking call
    return scores, selected_features
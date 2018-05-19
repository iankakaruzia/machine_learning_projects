import numpy as np
import libs.dist_metrics as dm

class KNNClassifier(object):
    
    def __init__(self, k=5, metric='minkowski', p=2):
        self.k = k
        self.metric = metric
        self.p = p
    
    def fit(self, X, y):
        self.y_ = y.reshape(len(y), 1)
        if self.metric == 'minkowski':
            self.distances = dm.minkowski_distance(X, self.y_, self.p)
        self.idx_sort = np.argsort(self.distances)
        return self.idx_sort[1:self.k+1]
    
    def predict(self, X):
        self.output_values = self.y_[self.idx_sort]
        self.counts = np.unique(self.output_values, return_counts=True)
        self.idx_max = np.argmax(self.counts[1])
        self.prediction = self.counts[0][self.idx_max]
        return self.prediction
    
class KNNRegressor(object):
    
    def __init__(self, k=5, metric='minkowski', p=2):
        self.k = k
        self.metric = metric
        self.p = p
    
    def fit(self, X, y):
        self.y_ = y.reshape(len(y), 1)
        if self.metric == 'minkowski':
            self.distances = dm.minkowski_distance(X, self.y_, self.p)
        self.idx_sort = np.argsort(self.distances)
        return self.idx_sort[1:self.k+1]
    
    def predict(self, X):
        self.output_values = self.y_[self.idx_sort]
        self.prediction = np.sum(self.output_values) / self.output_values.shape[0]
        return self.prediction
        
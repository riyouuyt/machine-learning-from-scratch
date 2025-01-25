import numpy as np

class LinearRegression:
    '''
    important step is:
        Training
        1. Initialize W as zero
        2. Initialize B as zero

    given a data point:
        1. predict result by using y = wx + b
        2. calculate error
        3. use gradient descent to figure out new weight and bias
        4. repeat number times

     testing
    '''
    def __init__(self, learning_rate=0.01, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
    
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iters):
            y_pred = np.dot(X, self.weights) + self.bias
        
            dw = (1/n_samples) * np.dot(X.T, (y_pred - y))
            db = (1/n_samples) * np.sum(y_pred - y)

            self.weight = self.weight - self.lr * dw
            self.bias = self.bias - self.lr * db
            
    def predict(self, X):
        y_pred = np.dot(X, self.weights) + self.bias
        return y_pred
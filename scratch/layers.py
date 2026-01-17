import numpy as np

class Linear:
    def __init__(self, in_features, out_features):
        self.W = np.random.randn(in_features, out_features) * 0.01
        self.b = np.zeros(out_features)

        # will be failed during forward/backward
        self.X = None
        self.dW = None
        self.db = None

    def forward(self, X):
        self.X = X #cache for backward
        return X @ self.W + self.b

    def backward(self, dZ):
        self.dW = self.X.T @ dZ
        self.db = np.sum(dZ, axis=0)
        dX = dZ @ self.W.T
        return dX

class ReLU:
    def __init__(self):
        self.X = None
        
    def forward(self, X):
        self.X = X
        return np.maximum(0, X)

    def backward(self, dA):
        dX = dA.copy()
        dX[self.X <= 0] = 0
        return dX



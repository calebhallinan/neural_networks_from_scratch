"""
Adam optimizer implementation.
"""
# Nummpy
import numpy as np

class Adam:
    """
    Adam optimizer class.
    """
    def __init__(self, params, lr=1e-3):
        self.params = params
        self.lr = lr
        self.m = [np.zeros_like(p) for p in params]
        self.v = [np.zeros_like(p) for p in params]
        self.t = 0

    def step(self, grads):
        """
        Performs a single optimization step.
        """
        self.t += 1
        for i,(p,g) in enumerate(zip(self.params, grads)):
            self.m[i] = 0.9*self.m[i] + 0.1*g
            self.v[i] = 0.999*self.v[i] + 0.001*(g**2)
            p -= self.lr * self.m[i] / (np.sqrt(self.v[i]) + 1e-8)

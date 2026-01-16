"""
Loss Funtion
"""

import numpy as np

class SoftmaxCrossEntropyLoss:
    """
    Softmax Cross-Entropy Loss
    """
    def __init__(self):
        self.y = None
        self.p = None

    def forward(self, logits, y):
        """
        Compute the softmax cross-entropy loss.
        """
        self.y = y
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        self.p = exp / exp.sum(axis=1, keepdims=True)
        return -np.mean(np.log(self.p[np.arange(len(y)), y]))

    def backward(self):
        """
        Compute the gradient of the loss with respect to logits.
        """
        if self.p is None or self.y is None:
            raise ValueError("forward() must be called before backward()")
        grad = self.p.copy()
        grad[np.arange(len(self.y)), self.y] -= 1
        return grad / len(self.y)

from abc import ABC, abstractmethod
import numpy as np

# Base class for loss functions
class LossFunction(ABC):

    def __init__(self):
        self.grad = None

    @abstractmethod
    def get_loss(self, in_data):
        pass

    def get_grad(self):
        return self.grad


class MeanSquaredError(LossFunction):

    def get_loss(self, probs, pred):
        delta = probs-pred
        self.grad = delta
        per_sample_loss = .5*np.sum(np.square(delta), axis=0)
        return np.mean(per_sample_loss)


# Automatically uses softmax
class SoftMaxCrossEntropy(LossFunction):

    def get_loss(self, logits, vals):
        probs = self.SoftMax(logits)
        probs = np.clip(probs, 1e-12, 1-1e-12)
        loss = -np.mean(np.sum(np.log(probs)*vals, axis=0))
        self.grad = probs-vals
        return loss


    def SoftMax(self, logits):
        logits = logits - np.max(logits, axis=0, keepdims=True)
        exps = np.exp(logits)
        col_sums =  np.sum(exps, axis=0, keepdims=True)
        return exps/col_sums


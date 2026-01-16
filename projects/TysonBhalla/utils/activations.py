from abc import ABC, abstractmethod
import numpy as np


# Base class for activation functions
class ActivationFunction(ABC):

    def __init__(self):
        self.grad = None


    @abstractmethod
    def forward(self, in_data):
        return in_data

    def back(self, prev_grad, batch_size):
        return prev_grad*self.grad

    def get_grad(self):
        return self.grad

    def update(self, learning_rate):
        pass

    def __str__(self):
        return type(self).__name__

    def type(name):
        return {"Blank": Blank(),
                "ReLU": ReLU(),
                "SoftMax": SoftMax()}[name]

    


# Identity function
class Blank(ActivationFunction):
    def forward(self, in_data):
        self.grad = np.ones(in_data.shape)
        return in_data


class ReLU(ActivationFunction):
    def forward(self, in_data):
        out = np.maximum(0,in_data)
        self.grad = 1*(out!=0)
        return out

class SoftMax(ActivationFunction):
    def forward(self, logits):
        logits_shifted = logits - np.max(logits, axis=0, keepdims=True)
        exps = np.exp(logits_shifted)
        softmax = exps / np.sum(exps, axis=0, keepdims=True)
        
        # Calculate jacobians for gradient calculation
        self.grad = []
        for col in softmax.T:
            self.grad.append(np.identity(col.size)*col - np.outer(col, col.T))

        return softmax

    def back(self, prev_grad, batch_size):
        new_grad = np.zeros(prev_grad.shape)

        # Multiply each sample with its respective jacobian matrix
        for i in range(batch_size):
            new_grad[:,i] = (self.grad[i]@prev_grad[:,i])
        return new_grad
            
# Not done
class Sigmoid(ActivationFunction):
    pass

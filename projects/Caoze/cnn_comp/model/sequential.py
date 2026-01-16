"""
Model Container/Neural Network Sequential Model
Network architecture is defined here in a sequential manner.
"""

class SequentialModel:
    """A simple sequential model to stack layers."""
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        """
        Define the forward pass through all layers
        """
        for l in self.layers:
            x = l.forward(x)
        return x

    def backward(self, grad):
        """
        Define the backward pass through all layers
        """
        for l in reversed(self.layers):
            grad = l.backward(grad)
            assert grad is not None, f"{l.__class__.__name__}.backward returned None"

    def params(self):
        """
        Return all parameters of the model
        """
        return [p for l in self.layers for p in l.params()]

    def grads(self):
        """
        Return all gradients of the model
        """
        return [g for l in self.layers for g in l.grads()]

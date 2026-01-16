"""
Stochastic Gradient Descent (SGD) optimizer implementation.
"""
class SGD:
    """
    Stochastic Gradient Descent optimizer class.
    """
    def __init__(self, learning_rate=0.01):
        self.parameters = []
        self.learning_rate = learning_rate

    def step(self, layer):
        """
        Update the parameters of the given layer using its gradients.
        """
        for p, g in zip(self.parameters, layer.gradients()):
            p -= self.learning_rate * g

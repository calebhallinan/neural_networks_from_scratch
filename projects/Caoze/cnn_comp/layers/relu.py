"""
    Contains ReLU activation layer implementation.
    And Leaky ReLU variant.
"""
import numpy as np

class ReLU:
    """
        ReLU activation layer.
    """
    def __init__(self):
        self.mask = None

    def forward(self, input_data):
        """
            Forward pass of ReLU activation.
        """
        assert not np.isnan(input_data).any()
        self.mask = input_data > 0
        return input_data * self.mask

    def backward(self, output_gradient):
        """
            Backward pass of ReLU activation.
        """
        return output_gradient * self.mask

    def params(self):
        """Null Parameters method"""
        return []  # No parameters in ReLU layer

    def grads(self):
        """Null Gradients method"""
        return []   # No gradients in ReLU layer

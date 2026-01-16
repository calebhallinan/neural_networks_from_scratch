"""
Dense Layer Class
"""
import numpy as np

class Dense:
    """
    A fully connected dense layer for neural networks
    """
    def __init__(self, input_size, output_size):
        # Initialize weights and biases
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.biases = np.zeros((output_size))
        self.input_data = None
        self.grad_weights = None
        self.grad_biases = None

    def forward(self, input_data):
        """
        Compute the forward pass
        """
        assert not np.isnan(input_data).any()
        # Perform the forward pass
        self.input_data = input_data  # Store input for backpropagation
        forward_output = np.dot(input_data, self.weights) + self.biases
        return forward_output

    def backward(self, grad_output):
        """
        Compute gradients with respect to weights, biases, and input data
        """
        # Perform the backward pass
        if self.input_data is None:
            raise ValueError("backward() called before forward()")
        self.grad_weights = self.input_data.T @ grad_output
        self.grad_biases = grad_output.sum(axis=0)

        dx = grad_output @ self.weights.T
        return dx

    def params(self):
        """
        Return weights and biases
        """
        # Return weights and biases
        return [self.weights, self.biases]

    def grads(self):
        """
        Return gradients of weights and biases
        """
        # Return gradients of weights and biases
        return [self.grad_weights, self.grad_biases]

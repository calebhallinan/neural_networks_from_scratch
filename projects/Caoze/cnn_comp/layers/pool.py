""" 
    For pooling layers.
    No parameter & reduction in dimensionality.
    Route gradient to max index.
"""
import numpy as np

class MaxPool2D:
    """
        Max Pooling layer for 2D inputs.
    """
    def __init__(self, kernel_size, stride=2):
        self.k = kernel_size
        self.s = stride
        self.input = None
        self.argmax = {}  # To store the indices of max values

    def forward(self, input_data):
        """
            Forward pass of Max Pooling.
        """
        assert not np.isnan(input_data).any()
        self.input = input_data
        batch_size, channels, height, width = input_data.shape
        out_height = (height - self.k)// self.s + 1
        out_width = (width - self.k)// self.s + 1

        out = np.zeros((batch_size, channels, out_height, out_width))
        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_height):
                    for j in range(out_width):
                        h = i * self.s
                        w = j * self.s

                        patch = input_data[b, c, h:h+self.k, w:w+self.k]
                        out[b, c, i, j] = np.max(patch)
                        self.argmax[b,c,i,j] = np.argmax(patch)

        return out

    def backward(self, d_out):
        """
            Backward pass of Max Pooling.
        """
        batch_size, channels, height_out, width_out = d_out.shape
        dx = np.zeros_like(self.input)

        for b in range(batch_size):
            for c in range(channels):
                for i in range(height_out):
                    for j in range(width_out):
                        h = i * self.s
                        w = j * self.s
                        idx = self.argmax[b, c, i, j]
                        dh, dw = divmod(idx, self.k)
                        dx[b, c, h+dh, w+dw] += d_out[b, c, i, j]
        return dx

    def params(self):
        """Null Parameters method"""
        return []  # No parameters in pooling layer

    def grads(self):
        """Null Gradients method"""
        return []   # No gradients in pooling layer

"""
A convolutional layer for neural networks.
"""
import numpy as np

class Conv2D:
    """
    A 2D convolutional layer.
    """
    def __init__(self, in_c, out_c, k, stride=1, padding=0):
        self.stride = stride
        self.padding = padding
        self.k = k
        self.x = None
        self.x_pad = None

        scale = np.sqrt(2 / (in_c * k * k))
        self.W = np.random.randn(out_c, in_c, k, k) * scale
        self.b = np.zeros(out_c)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    def forward(self, x):
        """
        Forward pass of the convolutional layer.
        """
        assert x.shape[1] == self.W.shape[1], \
        f"Conv expected {self.W.shape[1]} channels, got {x.shape[1]}"
        
        self.x = x
        N, C, H, W = x.shape
        F, _, KH, KW = self.W.shape

        height_out = (H + 2*self.padding - KH) // self.stride + 1
        weight_out = (W + 2*self.padding - KW) // self.stride + 1

        x_pad = np.pad(x,
            ((0,0),(0,0),(self.padding,self.padding),(self.padding,self.padding))
        )
        self.x_pad = x_pad

        out = np.zeros((N, F, height_out, weight_out))

        for n in range(N):
            for f in range(F):
                for i in range(height_out):
                    for j in range(weight_out):
                        h = i * self.stride
                        w = j * self.stride
                        out[n,f,i,j] = np.sum(
                            x_pad[n,:,h:h+KH,w:w+KW] * self.W[f]
                        ) + self.b[f]

        return out

    def backward(self, dout):
        """
        Backward pass of the convolutional layer.
        """
        if self.x is None:
            raise RuntimeError("backward() called before forward()")
        if dout is None:
            raise RuntimeError("backward() called with None gradient")
        if self.x_pad is None:
            raise RuntimeError("backward() called before forward()")
        N, C, H, W = self.x.shape
        F, _, KH, KW = self.W.shape

        dx = np.zeros_like(self.x_pad)
        self.dW = np.zeros_like(self.W)
        self.db = np.sum(dout, axis=(0,2,3))

        height_out, weight_out = dout.shape[2:]

        for n in range(N):
            for f in range(F):
                for i in range(height_out):
                    for j in range(weight_out):
                        h = i * self.stride
                        w = j * self.stride
                        dx[n,:,h:h+KH,w:w+KW] += self.W[f] * dout[n,f,i,j]
                        self.dW[f] += self.x_pad[n,:,h:h+KH,w:w+KW] * dout[n,f,i,j]

        if self.padding > 0:
            return dx[:,:,self.padding:-self.padding,self.padding:-self.padding]
        else:
            return dx

    def params(self):
        """
        return weights and biases
        """
        return [self.W, self.b]
    def grads(self):
        """
        return gradients of weights and biases
        """
        return [self.dW, self.db]

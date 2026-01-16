from utils import activations
import numpy as np

class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def back_prop(self, grad, batch_size):
        for layer in self.layers[::-1]:
            grad = layer.back(grad, batch_size)

    def update(self, learning_rate):
        for layer in self.layers:
            layer.update(learning_rate)

    def add(self, layer):
        self.layers.append(layer)

    def __str__(self):
        output = "MODEL:\n"
        for layer in self.layers:
            output += str(layer)+"\n"
        return output.rstrip('\n')

    def save(self, filename):
        out_layers = []
        for layer in self.layers:
            if isinstance(layer, Layer):
                out_layers.append({
                    "type": "linear",
                    "shape": (layer.in_count, layer.out_count),
                    "weights": layer.weights,
                    "bias": layer.bias
                })
            elif isinstance(layer, activations.ActivationFunction):
                out_layers.append({
                    "type": "activation",
                    "class": str(layer)
                })
        np.savez_compressed(filename, *out_layers)

    def load(filename):
        data = np.load(filename, allow_pickle=True)
        model = NeuralNetwork()
        for key in sorted(data.keys()):
            layer_info = data[key].item()
            if layer_info["type"] == "linear":
                model.add(Layer(*layer_info["shape"],
                                weights=layer_info["weights"],
                                bias=layer_info["bias"]))
            elif layer_info["type"] == "activation":
                model.add(activations.ActivationFunction.type(layer_info["class"]))
        return model




class Layer:

    def __init__(self, in_count, out_count, weights=None, bias=None):
        self.in_count = in_count
        self.out_count = out_count
        self.weights = self.randomize_weights(in_count, out_count) if weights is None else weights 
        self.bias = np.zeros((out_count,1)) if bias is None else bias

        self.last_inp = None
        self.dW = None
        self.db = None

    
    def randomize_weights(self, in_count, out_count):
        # He Initialization
        scale=np.sqrt(2/in_count)
        size=(out_count,in_count)
        return np.random.normal(loc=0, scale=scale, size=size)


    def forward(self, x):
        self.last_inp = x
        return np.matmul(self.weights,x)+self.bias

    def back(self, prev_grad, batch_size):
        self.dW = 1/batch_size * prev_grad@self.last_inp.T
        self.db = 1/batch_size * np.sum(prev_grad, axis=1, keepdims=True)
        return self.weights.T@prev_grad

    def update(self, learning_rate):
        self.weights -= learning_rate*self.dW
        self.bias -= learning_rate*self.db

    def __str__(self):
        return f"Linear: {self.in_count} -> {self.out_count}"

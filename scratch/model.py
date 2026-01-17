import numpy as np
from layers import Linear, ReLU

class MLP:
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1= Linear(input_size, hidden_size)
        self.act1= ReLU()
        self.fc2= Linear(hidden_size, output_size)

    def forward(self, X):
        z1= self.fc1.forward(X)
        a1= self.act1.forward(z1)
        scores= self.fc2.forward(a1)
        return scores

    def backward(self, dScores):
        dA1= self.fc2.backward(dScores)
        dZ1= self.act1.backward(dA1)
        dX= self.fc1.backward(dZ1)
        return dX

    def step(self, lr):
        self.fc1.W -= lr * self.fc1.dW
        self.fc1.b -= lr * self.fc1.db
        self.fc2.W -= lr * self.fc2.dW
        self.fc2.b -= lr * self.fc2.db
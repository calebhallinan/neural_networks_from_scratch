"""
Holds training routines for the CNN model.
"""
import numpy as np

def train_model(model, data, labels, epochs, learning_rate):
    """Train the CNN model with the provided data and labels."""
    for epoch in range(epochs):
        # Forward pass
        predictions = model.forward(data)

        # Compute loss (assuming mean squared error for simplicity)
        loss = np.mean((predictions - labels) ** 2)
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss}")

        # Backward pass
        loss_gradient = 2 * (predictions - labels) / labels.size
        model.backward(loss_gradient, learning_rate)

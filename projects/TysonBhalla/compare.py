from utils import neuralnetwork as nn
from utils import activations, losses
from utils import dataloader
import numpy as np
import matplotlib.pyplot as plt


import torch
from torch import nn as NN
import torch.nn.init as init
import torchvision.transforms as transforms

TRAIN_IMG_PATH = "data/train-images.gz"
TRAIN_LBL_PATH = "data/train-labels.gz"
TEST_IMG_PATH = "data/test-images.gz"
TEST_LBL_PATH = "data/test-labels.gz"

MNIST_dataloader = dataloader.DataLoader(TRAIN_IMG_PATH, TRAIN_LBL_PATH, 
                                         TEST_IMG_PATH, TEST_LBL_PATH)

# My model creation
mymodel = nn.NeuralNetwork()
mymodel.layers = [nn.Layer(28*28,128),
                activations.ReLU(),
                nn.Layer(128,64),
                activations.ReLU(),
                nn.Layer(64,10)]

# PyTorch model creation
class NeuralNetwork(NN.Module):
    def __init__(self):
        super().__init__()
        self.flatten = NN.Flatten()
        self.linear_relu_stack = NN.Sequential(
            NN.Linear(28*28, 128),
            NN.ReLU(),
            NN.Linear(128, 64),
            NN.ReLU(),
            NN.Linear(64, 10)
        )
        self.apply(self.init_weights)

    # He initialization
    def init_weights(self, m):
        if isinstance(m, NN.Linear):
            init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits
ptmodel = NeuralNetwork()

# Hyperparameters
learning_rate = .005
batch_size = 64
epochs = 25

# Loss and optimizer functions
myloss_fn = losses.SoftMaxCrossEntropy()
loss_fn = NN.CrossEntropyLoss()
optimizer = torch.optim.SGD(ptmodel.parameters(), lr=learning_rate)

# Variables for Plotting
epoch_x = range(epochs)
my_loss_plot = []
my_acc_plot = []
pt_loss_plot = []
pt_acc_plot = []


# Epoch loops
for i in range(epochs):
    print(f"Epoch {i+1}: |", end="", flush=True)
    
    # Training
    my_loss = 0
    batch_count = 0
    pt_loss_total = 0
    ptmodel.train()
    for y,X in MNIST_dataloader.get_data(data="train", batch_size=batch_size):
        batch_count += 1

        #my model
        one_hot_vals = np.zeros((10, len(y)))
        one_hot_vals[y, np.arange(len(y))] = 1
        
        X_flat = X.reshape(X.shape[0],-1)/255
        preds = mymodel.forward(X_flat.T)
        loss_val = myloss_fn.get_loss(preds, one_hot_vals)
        my_loss += loss_val

        loss_grad = myloss_fn.get_grad()
        mymodel.back_prop(loss_grad, batch_size)
        mymodel.update(learning_rate)

        # pytorch model
        X = torch.tensor(X, dtype=torch.float32) / 255.0
        pred = ptmodel(X)
        pt_loss = loss_fn(pred, torch.tensor(y, dtype=torch.long))

        optimizer.zero_grad()
        pt_loss.backward()
        optimizer.step()
        pt_loss_total += pt_loss.item()

        # printing
        if batch_count%50 == 0:
            print("=", end="", flush=True)

    print("|")
    my_loss_plot.append(my_loss/batch_count)
    pt_loss_plot.append(pt_loss_total/batch_count)
    print(f" --Average Training Loss (my): {my_loss/batch_count:.4f}")
    print(f" --Average Training Loss (pt): {pt_loss_total/batch_count:.4f}")


    # Testing
    my_correct = 0
    pt_correct = 0
    count = 0
    ptmodel.eval()
    for y,X in MNIST_dataloader.get_data(data="test", batch_size=batch_size):
        # my model
        X_flat = X.reshape(X.shape[0],-1) / 255
        preds = mymodel.forward(X_flat.T)
        maxs = np.argmax(preds, 0)
        my_correct += sum(maxs==y)
        count += batch_size

        # pytorch model
        X = torch.tensor(X, dtype=torch.float32) / 255.0
        pred = ptmodel(X)
        y = torch.tensor(y, dtype=torch.long)
        pt_correct += (pred.argmax(1) == y).type(torch.float).sum().item()

        

    my_acc_plot.append(my_correct/count*100)
    pt_acc_plot.append(pt_correct/count*100)
    print(f" -- My Accuracy: {my_correct/count*100:.4f}%")
    print(f" -- PT Accuracy: {pt_correct/count*100:.4f}%")
    print()




# Plot accuracy and loss over epochs
plt.plot(epoch_x, my_acc_plot, 'b', label='My Model')
plt.plot(epoch_x, pt_acc_plot, 'r', label='PT Model')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Accuracy of Models Across Epochs')

plt.show()

plt.clf()

plt.plot(epoch_x, my_loss_plot, 'b', label='My Model')
plt.plot(epoch_x, pt_loss_plot, 'r', label='PT Model')
plt.xlabel('Epochs')
plt.ylabel('Average Loss')
plt.title('Average Loss of Models Across Epochs')

plt.show()



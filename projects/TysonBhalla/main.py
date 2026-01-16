from utils import neuralnetwork as nn
from utils import activations, losses
from utils import dataloader
import numpy as np
import matplotlib.pyplot as plt

TRAIN_IMG_PATH = "data/train-images.gz"
TRAIN_LBL_PATH = "data/train-labels.gz"
TEST_IMG_PATH = "data/test-images.gz"
TEST_LBL_PATH = "data/test-labels.gz"

MNIST_dataloader = dataloader.DataLoader(TRAIN_IMG_PATH, TRAIN_LBL_PATH, 
                                         TEST_IMG_PATH, TEST_LBL_PATH)

# Model definition
model = nn.NeuralNetwork()
model.layers = [nn.Layer(28*28,128),
                activations.ReLU(),
                nn.Layer(128,64),
                activations.ReLU(),
                nn.Layer(64,10)]#,
                #activations.SoftMax()]



#loss_fn = losses.MeanSquaredError()
loss_fn = losses.SoftMaxCrossEntropy()
learning_rate = .005
batch_size = 64
epochs = 10

#model = nn.NeuralNetwork.load("200epochmodel.npz")
#print(model)

# Variables for Plotting
epoch_x = range(epochs)
loss_plot = []
acc_plot = []

# Epoch loop
for i in range(epochs):
    print(f"Epoch {i+1}: |", end="", flush=True)
    
    # Training
    loss = 0
    batch_count = 0
    for y,X in MNIST_dataloader.get_data(data="train", batch_size=batch_size):
        batch_count += 1
        one_hot_vals = np.zeros((10, len(y)))
        one_hot_vals[y, np.arange(len(y))] = 1
        
        X = X.reshape(X.shape[0],-1)/255
        preds = model.forward(X.T)
        loss_val = loss_fn.get_loss(preds, one_hot_vals)
        loss += loss_val

        loss_grad = loss_fn.get_grad()
        model.back_prop(loss_grad, batch_size)
        model.update(learning_rate)

        if batch_count%50 == 0:
            print("=", end="", flush=True)

    print("|")
    print(f" --Average Training Loss: {loss/batch_count:.4f}")
    loss_plot.append(loss/batch_count)


    # Testing
    correct = 0
    count = 0
    for y,X in MNIST_dataloader.get_data(data="test", batch_size=batch_size):
        X = X.reshape(X.shape[0],-1) / 255
        preds = model.forward(X.T)
        maxs = np.argmax(preds, 0)
        correct += sum(maxs==y)
        count += batch_size

    print(f" --Accuracy: {correct/count*100:.4f}%")
    print()
    acc_plot.append(correct/count*100)


#model.save("300onceepochmodel.npz")


# Plot accuracy and loss over epochs
fig, ax1 = plt.subplots()

ax1.plot(epoch_x, acc_plot, 'b', label='Accuracy')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Accuracy', color='b')
ax1.tick_params(axis='y', labelcolor='b')

ax2 = ax1.twinx()
ax2.plot(epoch_x, loss_plot, 'r', label='Average Loss')
ax2.set_ylabel('Average Loss', color='r')
ax2.tick_params(axis='y', labelcolor='r')

plt.title('Training Accuracy and Loss per Epoch')
ax1.grid(True)

plt.show()

from utils import neuralnetwork as nn
from utils import activations, losses
from utils import dataloader
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
import time

TRAIN_IMG_PATH = "data/train-images.gz"
TRAIN_LBL_PATH = "data/train-labels.gz"
TEST_IMG_PATH = "data/test-images.gz"
TEST_LBL_PATH = "data/test-labels.gz"

MNIST_dataloader = dataloader.DataLoader(TRAIN_IMG_PATH, TRAIN_LBL_PATH, 
                                         TEST_IMG_PATH, TEST_LBL_PATH)

num_batch_exps = 10
starting_exp = 4
learning_rate = .005
epochs = 10
loss_batch = []
acc_batch = []
time_batch = []

model = nn.NeuralNetwork()
model.layers = [nn.Layer(28*28,128),
                activations.ReLU(),
                nn.Layer(128,64),
                activations.ReLU(),
                nn.Layer(64,10)]#,
                #activations.SoftMax()]
model.save("batchtestmodel.npz")

for exp in range(starting_exp, num_batch_exps+starting_exp):
    batch_size = np.power(2,exp)
    print(batch_size)
    start = time.perf_counter()

    # getting same model per batch
    model = nn.NeuralNetwork.load("batchtestmodel.npz")

    #loss_fn = losses.MeanSquaredError()
    loss_fn = losses.SoftMaxCrossEntropy()


    # Variables for Plotting
    loss_plot = []
    acc_plot = []

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

    end = time.perf_counter()

    loss_batch.append(loss_plot)
    acc_batch.append(acc_plot)
    time_batch.append(end-start)



# Plot accuracy and loss over epochs

fig, ax = plt.subplots()
cmap = plt.cm.viridis
norm = plt.Normalize(starting_exp, num_batch_exps+starting_exp-1)
ax.set_xlabel('Epochs')
ax.set_ylabel('Accuracy')
ax.set_title('Accuracy for Each Epoch and Batch Size')

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig.colorbar(sm, ax=ax, label="Exponent e (batch size = $2^e$)")


for i in range(num_batch_exps):
    ax.plot(range(epochs+1), acc_batch[i], color=cmap(norm(i+starting_exp)))

plt.show()
plt.clf()


fig, ax = plt.subplots()
cmap = plt.cm.viridis
norm = plt.Normalize(starting_exp, num_batch_exps+starting_exp-1)

ax.set_xlabel('Epochs')
ax.set_ylabel('Average Batch Lass')
ax.set_title('Average Batch Loss for Each Epoch and Batch Size')

sm = ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])

fig.colorbar(sm, ax=ax, label="Exponent e (batch size = $2^e$)")

for i in range(num_batch_exps):
    ax.plot(range(1,epochs+1), loss_batch[i], color=cmap(norm(i+starting_exp)))

plt.show()
plt.clf()
    
plt.plot(range(starting_exp, num_batch_exps+starting_exp), time_batch)
plt.xlabel('Batch Size e^x')
plt.ylabel('Time for 10 epochs')
plt.title('Batch Size vs Time')
plt.show()

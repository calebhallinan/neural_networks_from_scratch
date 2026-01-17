import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)
X=np.random.rand(200,2)
y = (X[:, 0] > np.median(X[:, 0])).astype(int)
# print("Unique classes in y:", np.unique(y))
# print("Counts:", np.bincount(y))
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Class 1")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Toy Dataset")
plt.show()

# from layers import Linear

# layer=Linear (in_features=2, out_features=3)
# Z=layer.forward(X)
# print("X shape:", X.shape)
# print("W shape:", layer.W.shape)
# print("b shape:", layer.b.shape)
# print("Z shape:", Z.shape)

# from layers import ReLU

# relu=ReLU()
# A= relu.forward(Z)
# print("A shape:", A.shape)
# print("Number of negative values in A:", np.sum(A<0))

from model import MLP

np.random.seed(0)

net= MLP(input_size=2, hidden_size=8, output_size=2)
# scores=net.forward(X)

# print("Scores shape:", scores.shape)

# from losses import softmax, cross_entropy_loss

# probs = softmax(scores)
# loss = cross_entropy_loss(probs, y)

# print("probs shape:", probs.shape)
# print("first row probs:", probs[0])
# print("loss:", loss)

from losses import softmax, cross_entropy_loss, softmax_cross_entropy_backward
lr = 0.5
epochs = 100

loss_history = []

# print("W1 mean before:", net.fc1.W.mean())
# print("W2 mean before:", net.fc2.W.mean())


for epoch in range(epochs):
    scores = net.forward(X)
    probs = softmax(scores)
    loss = cross_entropy_loss(probs, y)

    loss_history.append(loss)

    dScores = softmax_cross_entropy_backward(probs, y)
    net.backward(dScores)
    net.step(lr)

    # if epoch in [0, 1]:
        # print("W1 abs-mean:", np.mean(np.abs(net.fc1.W)), "W2 abs-mean:", np.mean(np.abs(net.fc2.W)))
        # print("dW1 abs-mean:", np.mean(np.abs(net.fc1.dW)), "dW2 abs-mean:", np.mean(np.abs(net.fc2.dW)))

    if epoch % 10 == 0:
        preds = np.argmax(probs, axis=1)
        acc = np.mean(preds == y)
        print(f"epoch {epoch:03d} | loss {loss:.4f} | acc {acc:.3f}")

# Final training accuracy
scores = net.forward(X)
probs = softmax(scores)
preds = np.argmax(probs, axis=1)
acc = np.mean(preds == y)
print (f"Final training accuracy:", acc)

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.show()

def plot_decision_boundary(model, X, y, steps=200):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, steps),
        np.linspace(y_min, y_max, steps)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]  # (steps*steps, 2)

    scores = model.forward(grid)
    probs = softmax(scores)
    preds = np.argmax(probs, axis=1).reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, preds, alpha=0.3)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Class 1")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision Boundary")
    plt.show()

plot_decision_boundary(net, X, y)
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

# Reproducibility
np.random.seed(0)
torch.manual_seed(0)

# Same dataset as scratch version
X = np.random.rand(200, 2).astype(np.float32)
y = (X[:, 0] > np.median(X[:, 0])).astype(np.int64)

plt.figure()
plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0")
plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Class 1")
plt.legend()
plt.xlabel("x1")
plt.ylabel("x2")
plt.title("Toy Dataset (PyTorch)")
plt.show()


X_t = torch.from_numpy(X).float()
y_t = torch.from_numpy(y).long()

# Model: 2 -> 8 -> 2
model = nn.Sequential(
    nn.Linear(2, 8),
    nn.ReLU(),
    nn.Linear(8, 2)
)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.5)

epochs = 100
loss_history = []

for epoch in range(epochs):
    optimizer.zero_grad()

    scores = model(X_t)
    loss = criterion(scores, y_t)

    loss.backward()
    optimizer.step()

    loss_history.append(loss.item())

    if epoch % 10 == 0:
        preds = torch.argmax(scores, dim=1)
        acc = (preds == y_t).float().mean().item()
        print(f"epoch {epoch:03d} | loss {loss.item():.4f} | acc {acc:.3f}")

# Final accuracy
with torch.no_grad():
    scores = model(X_t)
    preds = torch.argmax(scores, dim=1)
    acc = (preds == y_t).float().mean().item()
    print("Final training accuracy:", acc)

# Loss curve
plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PyTorch Training Loss")
plt.show()

def plot_decision_boundary_torch(model, X, y, steps=200):
    x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
    y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, steps),
        np.linspace(y_min, y_max, steps)
    )

    grid = np.c_[xx.ravel(), yy.ravel()].astype(np.float32)
    grid_t = torch.from_numpy(grid)

    with torch.no_grad():
        scores = model(grid_t)
        preds = torch.argmax(scores, dim=1).numpy()

    preds = preds.reshape(xx.shape)

    plt.figure()
    plt.contourf(xx, yy, preds, alpha=0.3)

    plt.scatter(X[y == 0, 0], X[y == 0, 1], color="blue", label="Class 0")
    plt.scatter(X[y == 1, 0], X[y == 1, 1], color="red", label="Class 1")
    plt.legend()
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("PyTorch Decision Boundary")
    plt.show()

plt.figure()
plt.plot(loss_history)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("PyTorch Training Loss")
plt.show()

plot_decision_boundary_torch(model, X, y)



"""" 
Unpack MNIST dataset from raw ubyte files and save as a compressed .npz file.
"""
# Import necessary libraries
import struct
from pathlib import Path
import numpy as np

def load_images(path):
    """
    Load MNIST images from the given ubyte file.
    Returns a numpy array of shape (num_images, 28, 28).
    """
    with open(path, 'rb') as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        assert magic == 2051
        data = np.frombuffer(f.read(), dtype=np.uint8)
        images = data.reshape(n, rows, cols)
        return images

def load_labels(path):
    """
    Load MNIST labels from the given ubyte file.
    Returns a numpy array of shape (num_labels,).
    """
    with open(path, 'rb') as f:
        magic, n = struct.unpack(">II", f.read(8))
        assert magic == 2049
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

def main():
    """
    Main function to load MNIST data and save as compressed .npz file.
    """
    raw = Path(__file__).parent / "raw"

    x_train = load_images(raw / "train-images.idx3-ubyte")
    y_train = load_labels(raw / "train-labels.idx1-ubyte")
    x_test  = load_images(raw / "t10k-images.idx3-ubyte")
    y_test  = load_labels(raw / "t10k-labels.idx1-ubyte")

    # Normalize and reshape for CNN
    x_train = x_train.astype(np.float32) / 255.0
    x_test  = x_test.astype(np.float32) / 255.0

    x_train = x_train[:, None, :, :]   # (N, 1, 28, 28)
    x_test  = x_test[:, None, :, :]

    output_path = Path(__file__).parent / "mnist.npz"
    np.savez_compressed(
        output_path,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test
    )

    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()

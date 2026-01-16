"""
    Dataset classes, stores image  & label.
    Load images, resize and normalize them.
"""

class MNISTDataset:
    """Dataset class for MNIST images and labels."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class CIFARDataset:
    """Dataset class for CIFAR images and labels."""
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]

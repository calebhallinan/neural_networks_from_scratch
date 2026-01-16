"""
    Data loader module:
    Shuffles indices and yield mini-batches.
"""
# Numpy
import numpy as np

class DataLoader:
    """
    Data Loader for batching and shuffling data.
    """
    def __init__(self, data, batch_size, shuffle=True):
        self.dataset = data
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __iter__(self):
        """
        Yield mini-batches of data.
        """
        idx = np.arange(len(self.dataset))
        if self.shuffle:
            np.random.shuffle(idx)
        for i in range(0, len(idx), self.batch_size):
            batch = idx[i:i+self.batch_size]
            X, y = zip(*[self.dataset[j] for j in batch])
            yield np.array(X), np.array(y)

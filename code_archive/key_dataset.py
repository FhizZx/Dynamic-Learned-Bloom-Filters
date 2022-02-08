import torch
import numpy as np
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


# Dataset of keys (positive keys are those with label >= 0.5)
class KeyDataset(Dataset):

    def __init__(self, features, labels):
        n_keys = labels.size


        self.keys = np.asarray(range(n_keys)).astype(int)
        self.labels = labels
        self.pos_keys = np.asarray(([x for (x, y)
                                     in zip(self.keys, self.labels)
                                     if y >= 0.5])).astype(int)
        self.neg_keys = np.asarray(([x for (x, y)
                                     in zip(self.keys, self.labels)
                                     if y < 0.5])).astype(int)
        self.keys_t = torch.FloatTensor(self.keys).view(-1, 1)
        self.labels_t = torch.FloatTensor(labels).view(-1)
        self.mean = 0.5 * (n_keys - 1)
        self.std = math.sqrt((n_keys**2 - 1.0) / 12)

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        key = self.keys_t[idx]
        label = self.labels_t[idx]
        return key, label

    # Produce a new dataset with changed labels
    def changed_labels(self, idxs, changed_labels):
        labels = self.labels.copy()
        labels[idxs] = changed_labels
        return KeyDataset(labels)

    def plot(self, title="Key Dataset"):
        x = [self.pos_keys, self.neg_keys]
        color = ['g', 'r']
        green = mpatches.Patch(color='g', label='Positive Keys')
        red = mpatches.Patch(color='r', label='Negative Keys')
        plt.hist(x, bins=200, density=True, stacked=True, color=color)
        plt.title(title)
        plt.legend(handles=[green, red])
        plt.xlabel("Keys")
        plt.savefig("images/" + title)
        plt.clf()

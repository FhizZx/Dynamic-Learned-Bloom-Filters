import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


class BinaryDataset(Dataset):
    """ (pytorch friendly) dataset of items used for binary classification,
    more specifically in the context of a learned bloom filter.
        The labels are real numbers in [0, 1].
        The positive keys are the ones with label >= 0.5

        Attributes:
            keys -- used to identify data points (e.g in a BF)
            features -- extra information for training models
            labels -- whether an item is positive or negative
    """

    def __init__(self, keys, features, labels, standardize=False):
        self.keys = keys
        self.features = features
        if standardize:
            self.features = self.standardize(self.features)
        self.labels = labels
        self.pos_keys_idxs = np.flatnonzero(labels >= 0.5)
        self.neg_keys_idxs = np.flatnonzero(labels < 0.5)
        self.pos_keys = self.keys[self.pos_keys_idxs]
        self.neg_keys = self.keys[self.neg_keys_idxs]

        self.positives = self.pos_keys, self.features[self.pos_keys_idxs]
        self.negatives = self.neg_keys, self.features[self.neg_keys_idxs]

        self.features_t = torch.FloatTensor(self.features)
        self.labels_t = torch.FloatTensor(labels).view(-1)
        print("Created a dataset with {} positive and {} negative keys."
              .format(self.pos_keys.size, self.neg_keys.size))

    def __len__(self):
        return self.keys.size

    def __getitem__(self, idx):
        features = self.features_t[idx]
        label = self.labels_t[idx]
        return features, label

    def train_test_split(self, train_proportion):
        """ Split the dataset into train/test sets """
        train_size = int(len(self) * train_proportion)
        return self.random_split(train_size)

    def initial_added_split(self, initial_proportion, added_label=None):
        """ Split the dataset into a part that is added initially into a LBF,
        and a part that is added later.
            added_label = None, if the labels of the added dataset can be mixed
                        = 0, if the keys should be negative in the added set
                        = 1, if the keys should be positive in the addded set
        """
        initial_size = int(len(self) * initial_proportion)
        added_size = len(self) - initial_size

        if added_label is None:
            return self.random_split(initial_size)
        elif added_label == 0:
            added_idxs = np.random.choice(self.neg_keys_idxs,
                                          size=min(added_size,
                                                   self.neg_keys_idxs.size),
                                          replace=False)
        elif added_label == 1:
            added_idxs = np.random.choice(self.pos_keys_idxs,
                                          size=min(added_size,
                                                   self.pos_keys_idxs.size),
                                          replace=False)

        initial_idxs = np.setdiff1d(range(len(self)), added_idxs)
        return self.subset(initial_idxs), self.subset(added_idxs)

    def batches_split(self, batch_size):
        idxs = np.asarray(range(len(self)))
        np.random.shuffle(idxs)
        batch_datasets = []
        for i in range(0, len(self), batch_size):
            batch_idxs = idxs[i:max(i+batch_size, len(self))]
            batch_datasets.append(self.subset(batch_idxs))
        return batch_datasets

    def subset(self, idxs):
        return BinaryDataset(self.keys[idxs], self.features[idxs],
                             self.labels[idxs])

    def union(self, other):
        keys = np.hstack([self.keys, other.keys])
        features = np.vstack([self.features, other.features])
        labels = np.hstack([self.labels, other.labels])
        return BinaryDataset(keys, features, labels)

    def random_split(self, part_size):
        a_idxs = np.random.choice(range(len(self)), size=part_size,
                                  replace=False)
        b_idxs = np.setdiff1d(range(len(self)), a_idxs)
        return self.subset(a_idxs), self.subset(b_idxs)

    def plot(self, title="Key Dataset"):
        """ Plot the dataset according to the distribution of the keys"""
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

    @staticmethod
    def standardize(features):
        """ Make the mean 0, stddev 1 for the given sample set"""
        mean = features.mean(axis=0, keepdims=1)
        std = features.std(axis=0, keepdims=1)
        return (features - mean) / std

    @classmethod
    def from_csv(cls, path, standardize=False):
        """ Create a binary dataset from a csv file

            There should be one column for keys, one for labels and the rest
            should be for features intended for training a model (so numerical
            values)
        """
        df = pd.read_csv(path)  # pandas data frame
        keys = df['key'].to_numpy().flatten()
        features = df.drop(columns=['key', 'label']).values.astype(np.float32)
        labels = df['label'].to_numpy().astype(np.float32).flatten()
        return cls(keys, features, labels, standardize=standardize)

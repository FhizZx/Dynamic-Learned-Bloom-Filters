import matplotlib.pyplot as plt
import numpy as np
import matplotlib.patches as mpatches
from models.binary_logistic_nn import plot_model, np_to_pt, process_data, BinaryLogisticNN
import random
from utils.fisher_utils import apply_big_block_fisher_removal_update
import torch
import copy


def plot_data(pos_keys, neg_keys, added_elems):
    xplotted = [pos_keys, added_elems, neg_keys]
    colours = ['g', 'yellow', 'r']
    plt.hist(xplotted, bins=150, density=True, stacked=True, color=colours)
    green = mpatches.Patch(color='g', label='Positive Keys')
    red = mpatches.Patch(color='r', label='Negative Keys')
    yellow = mpatches.Patch(color='yellow', label='Newly Added Keys')
    plt.title("Key Distribution")
    plt.legend(handles=[green, red, yellow])
    plt.xlabel("Keys")
    plt.savefig('Data Histogram')
    plt.rcParams["figure.figsize"] = (6, 4)
    plt.clf()

# Generate some testing data


def gen_data(noise, added_elems):
    t1 = 1 - noise
    t2 = 1 - 10*noise
    t3 = 1 - 3*noise
    p1 = random.sample(range(10, 40), int(30*t1))
    p2 = random.sample(range(150, 200), int(50*t2))
    p3 = random.sample(range(500, 1000), int(50*t3))
    p4 = np.random.normal(1250, 20, int(350*t2))
    p4 = np.unique(np.floor(p4))
    p = p1 + p2 + p3 + p4.tolist()
    pos_keys = np.unique(np.asarray(p))
    neg_keys = np.asarray(list(set(range(0, 1500)) - set(p)))
    pos_keys_new = np.asarray(list(set(pos_keys.flatten()) | set(added_elems)))
    neg_keys_new = np.asarray(list(set(neg_keys.flatten()) - set(added_elems)))
    plot_data(pos_keys, neg_keys_new, added_elems)
    return pos_keys, neg_keys, pos_keys_new, neg_keys_new


added_elems = range(300, 500)
pos_keys, neg_keys, pos_keys_new, neg_keys_new = gen_data(
    0.05, added_elems)

# Train initial model
x_train, y_train, x_test, mean, std = process_data(pos_keys, neg_keys, 0.8)
model = BinaryLogisticNN(hidden_size=200, mean=mean, std=std)
model.do_train(x_train, y_train, 200000, lr=0.1, step_size=1000, decay=0.99)
plot_model(model, pos_keys, neg_keys, 0.0, "Initial trained model")

# Retrain model completely with newly added keys
x_train, y_train, x_test, mean, std = process_data(
    pos_keys_new, neg_keys_new, 0.8)
model_retrained = BinaryLogisticNN(hidden_size=200, mean=mean, std=std)
model_retrained.do_train(x_train, y_train, 200000,
                         lr=0.1, step_size=1000, decay=0.99)
plot_model(model_retrained, pos_keys_new, neg_keys_new, 0.0,
           "Completely retrained model with added keys")


# Train initial model on newly added keys
model_online = copy.deepcopy(model)
x_train_online = np.asarray(added_elems)
y_train_online = np.ones(x_train_online.size)
model_online.do_train(x_train_online, y_train_online, 300, 0.001, 10, 0.999)
plot_model(model_online, pos_keys_new, neg_keys_new, 0.0,
           "Online training on initial model with new keys")


# Make torch dataset out of keys
xd = np.hstack([pos_keys, neg_keys])
yd = np.hstack([np.ones(pos_keys.size), np.zeros(neg_keys.size)])
p = xd.argsort()
x, y = xd[p], yd[p]
x = np_to_pt(xd)
y = np_to_pt(yd)
y = torch.reshape(y, (-1,))
dataset = torch.utils.data.TensorDataset(x, y)


# SSSE to erase negative samples where new keys need to be added
no_params = 601
block_size = 601
batch_size = 20
eps = 3000.
damp = 0.1

model_ssse = apply_big_block_fisher_removal_update(model, dataset,
                                                   added_elems, damp,
                                                   batch_size, "cuda",
                                                   eps, block_size,
                                                   fisher_folder="D:\ISTernship\LBF-SSSE",
                                                   binary=True)


# Train SSSE-d model on newly added keys
model_online_ssse = copy.deepcopy(model_ssse)
x_train_online = np.asarray(added_elems)
y_train_online = np.ones(x_train_online.size)
model_online_ssse.do_train(
    x_train_online, y_train_online, 300, 0.001, 10, 0.999)
plot_model(model_online_ssse, pos_keys_new, neg_keys_new, 0.0,
           "Online training on SSSE'd model with new keys", legend=legend)

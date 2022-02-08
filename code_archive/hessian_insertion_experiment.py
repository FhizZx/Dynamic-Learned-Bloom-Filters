from utils.key_dataset import KeyDataset
from filters.online_LBF import OnlineLBF
from models.binary_logistic_nn import BinaryLogisticNN
import numpy as np
import os
import random
import imageio


def insertion_experiment(n_keys, init_pos_keys, added_keys, batch_size,
                         eps, title, train_noise=0.02):

    dir = './images/' + title + '/'
    if not os.path.exists(dir):
        os.makedirs(dir)
        os.makedirs(dir + 'dist/')
        os.makedirs(dir + 'metrics/')

    # Initialize labels as 0 + some noise
    labels = np.random.choice([0.0, 0.15, 0.25], n_keys,
                              p=[0.90, 0.05, 0.05])
    # labels = np.random.choice([0.0, 0.2], n_keys, p=[0.6, 0.4])
    labels[init_pos_keys] = 1.0
    dataset = KeyDataset(labels)

    # Train a model on the initial data and plot the result
    print("Loading initial model")
    model = BinaryLogisticNN(hidden_size=200,
                             mean=dataset.mean, std=dataset.std)

    if not os.path.exists("./cache/" + title + "_model.pt"):
        model.do_train(dataset, batch_size=2000, num_epochs=100000,
                       lr=0.11, step_size=3000, decay=0.95, noise=train_noise,
                       title="Initial Model")
        model.save(title + "_model.pt")
    else:
        model.load(title + "_model.pt")
    model.plot(dataset, title + " Initial Model")

    # Initialise the online LBF
    olbf = OnlineLBF(model, dataset, eps, 0.02)

    # Insert each batch
    batches = np.array_split(added_keys, added_keys.size/batch_size)
    random.Random(29051453).shuffle(batches)
    print(batches)
    for i, batch in enumerate(batches):
        print(title + " batch {:02d}".format(i))
        olbf.insert(batch, title + " batch {:02d}".format(i), dir,
                    momentum=1.0, span=30, n_samples=5*batch_size)

    # Create gif of insertions

    # make gif of distribution change
    images = []
    for file_name in sorted(os.listdir(dir + 'dist/')):
        file_path = os.path.join(dir + 'dist/', file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave('./images/'+title+'_dist.gif', images, fps=0.5)

    # make gif of metrics change
    images = []
    for file_name in sorted(os.listdir(dir + 'metrics/')):
        file_path = os.path.join(dir + 'metrics/', file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave('./images/'+title+'_metrics.gif', images, fps=0.5)


# Simple Ranges
def simple():
    n_keys = 2000
    pos_keys = set(range(20)) | set(range(500, 560)) | set(range(900, 1200)) | \
               set(range(1800, 2000))
    pos_keys = np.asarray(list(pos_keys)).astype(int)

    added_keys = set(range(200, 400)) | set(range(700, 900)) | \
                 set(range(1300, 1320))
    added_keys = np.asarray(list(added_keys)).astype(int)

    insertion_experiment(n_keys, pos_keys, added_keys, 30, 0.05, "Simple Range")


# Ranges with noise
def noisy1():
    n_keys = 2000
    pos_keys = set(range(20)) | set(range(500, 560)) | set(range(900, 1200)) | \
        set(range(1800, 2000))
    random_keys = set(random.choices(range(2000), k=100))
    pos_keys = pos_keys ^ random_keys
    pos_keys = np.asarray(list(pos_keys)).astype(int)

    added_keys = set(range(100, 400)) | set(range(700, 900)) | \
        set(range(1300, 1400))
    added_keys = np.asarray(list(added_keys))
    np.random.shuffle(added_keys)
    added_keys = np.sort(added_keys[:120:])

    insertion_experiment(n_keys, pos_keys, added_keys, 20, 0.05, "Noisy Range")


# Much more noise + normally distributed keys
def noisy2():
    n_keys = 2000
    p = np.random.normal(1250, 20, int(200))
    p = set(np.unique(np.floor(p)).astype(int).flatten())
    random_keys = set(random.choices(range(2000), k=600))
    pos_keys = (np.asarray(list(p ^ random_keys))).astype(int)
    neg_keys = list(set(range(2000)) - p - random_keys)
    added_keys = np.asarray(list(set(random.choices(neg_keys, k=400))))
    insertion_experiment(n_keys, pos_keys, added_keys, 20, 0.08, "Extra Noisy",
                         train_noise=0.08)


noisy1()

from utils.key_dataset import KeyDataset
from filters.online_LBF import OnlineLBF
from filters.online_LCBF import OnlineLCBF
from models.binary_logistic_nn import BinaryLogisticNN
import numpy as np
import os
import random
import imageio
import matplotlib

print(matplotlib.rcParams['font.family'])


def change_experiment(n_keys, init_pos_keys, batches, eps, title):
    # Initialize labels as 0 + some noise
    labels = np.random.choice([0.00, 0.1], n_keys, p=[0.6, 0.4])
    labels[range(80, 100)] = 0.0
    labels[100] = 1.0
    labels[139] = 1.0
    labels[range(140, 170)] = 0.0
    labels[init_pos_keys] = 0.98
    dataset = KeyDataset(labels)

    # Train a model on the initial data and plot the result
    print("Training model")
    model = BinaryLogisticNN(hidden_size=200,
                             mean=dataset.mean, std=dataset.std)

    if not os.path.exists("./cache/" + title + "_model.pt"):
        model.do_train(dataset, batch_size=2000, num_epochs=50000,
                       lr=0.12, step_size=1000, decay=0.98, noise=0.1,
                       title="Initial Model")
        model.save(title + "_model.pt")
    else:
        model.load(title + "_model.pt")
    model.plot(dataset, title + " Initial Model")

    # Initialise the online LBF
    olbf = OnlineLCBF(model, dataset, eps)

    for i, (batch, label) in enumerate(batches):
        print(title + " batch {}".format(i))
        olbf.change(batch, label, title + " batch {}".format(i))

    title = "Simple Range"
    # Create gif of insertions
    dir = './images/gifs'
    images = []
    for file_name in sorted(os.listdir(dir)):
        file_path = os.path.join(dir, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave('./images/'+title+'.gif', images, fps=0.2)


n_keys = 2000
pos_keys = set(range(100, 140)) | set(range(1000, 1200)) | \
           set(range(1800, 2000))
pos_keys = np.asarray(list(pos_keys)).astype(int)

added_keys = set(range(200, 600)) | set(range(700, 1000)) | \
             set(range(1300, 1700))
added_keys = np.asarray(list(added_keys)).astype(int)

batch1 = (np.asarray(range(200, 400)), 0.99)
batch2 = (np.asarray(range(700, 850)), 0.99)
batch3 = (np.asarray(range(100, 140)), 0.01)
batch4 = (np.asarray(range(730, 800)), 0.01)
batches = [batch1, batch2, batch3, batch4]

change_experiment(n_keys, pos_keys, batches, 0.2, "Simple Range")


def insertion_experiment(n_keys, init_pos_keys, added_keys, batch_size,
                         eps, title):
    # Initialize labels as 0 + some noise
    labels = np.random.choice([0.0, 0.2], n_keys, p=[0.6, 0.4])
    labels[init_pos_keys] = 1.0
    dataset = KeyDataset(labels)

    # Train a model on the initial data and plot the result
    print("Training model")
    model = BinaryLogisticNN(hidden_size=50,
                             mean=dataset.mean, std=dataset.std)

    if not os.path.exists("./cache/" + title + "_model.pt"):
        model.do_train(dataset, batch_size=2000, num_epochs=30000,
                       lr=0.12, step_size=2000, decay=0.99, noise=0.05,
                       title="Initial Model")
        model.save(title + "_model.pt")
    else:
        model.load(title + "_model.pt")
    model.plot(dataset, title + " Initial Model")

    # Initialise the online LBF
    olbf = OnlineLBF(model, dataset, eps)

    # Insert each batch
    batches = np.array_split(added_keys, n_keys/batch_size)
    random.Random(29051453).shuffle(batches)
    print(batches)
    for i, batch in enumerate(batches):
        print(title + " batch {}".format(i))
        olbf.insert(batch, title + " batch {}".format(i))

    title="Simple Range"
    # Create gif of insertions
    dir = './images/gifs'
    images = []
    for file_name in sorted(os.listdir(dir)):
        file_path = os.path.join(dir, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave('./images/'+title+'.gif', images, fps=0.2)

'''
n_keys = 2000
pos_keys = set(range(100, 140)) | set(range(1000, 1200)) | \
           set(range(1800, 2000))
pos_keys = np.asarray(list(pos_keys)).astype(int)

added_keys = set(range(200, 600)) | set(range(700, 1000)) | \
             set(range(1300, 1700))
added_keys = np.asarray(list(added_keys)).astype(int)

insertion_experiment(n_keys, pos_keys, added_keys, 300, 0.2, "Simple Range")'''

'''
labels = np.random.choice([0.0, 0.1], n_keys, p=[0.7, 0.3])
labels[pos_keys] = 1.0

dataset = KeyDataset(labels)

# Train a model on the initial data
print("Training model")
model = BinaryLogisticNN(hidden_size=200, mean=dataset.mean, std=dataset.std)

if not os.path.exists("./cache/model.pt"):
    model.do_train(dataset, batch_size=2000, num_epochs=100000,
                   lr=0.12, step_size=2000, decay=0.99, noise=0.05,
                   title="Initial Model")
    model.save("model.pt")
else:
    model.load("model.pt")

model.plot(dataset, "Initial Model")

# Initialise the online LBF
olbf = OnlineLBF(model, dataset, eps=0.2)

# Insert some elements in the online LBF
added_elems = np.asarray(list(range(300, 400)))
olbf.insert(added_elems)

# Insert some more elements in the online LBF
added_elems = np.asarray(list(range(400, 500)))
olbf.insert(added_elems)'''

'''dir = './images/gifs'
images = []
for file_name in sorted(os.listdir(dir)):
    file_path = os.path.join(dir, file_name)
    images.append(imageio.imread(file_path))
imageio.mimsave('./images/one_key_noisy.gif', images)'''

from utils.key_dataset import KeyDataset
from models.one_layer_mse_nn import OneLayerMSENN
from utils.single_step_updates import ihgp_update
import numpy as np
import os
import torch

# Initialize labels as 0 + some noise
labels = np.random.choice([0.0, 0.05, 0.35], 2000,
                          p=[0.90, 0.05, 0.05])
pos_keys = set(range(20)) | set(range(500, 560)) | set(range(900, 1200)) | \
           set(range(1800, 2000))
pos_keys = np.asarray(list(pos_keys)).astype(int)

labels[pos_keys] = 1.0
dataset = KeyDataset(labels)

# Train a model on the initial data and plot the result
print("Loading initial model")
model = OneLayerMSENN(hidden_size=200,
                       mean=dataset.mean, std=dataset.std)

if not os.path.exists("./cache/" + "l2_model.pt"):
    model.do_train(dataset, batch_size=2000, num_epochs=100000, lr=0.05,
                   weight_decay=0.0001, step_size=3000, decay=0.93, noise=0.05,
                   title="Initial Model")
    model.save("l2_model.pt")
else:
    model.load("l2_model.pt")
model.plot(dataset, "L2 Model")

# Do ihgp update
added_keys = set(range(200, 400))
added_keys = np.asarray(list(added_keys)).astype(int)

updated_model = ihgp_update(model, dataset, added_keys, 1.0,
                            "cuda", eps_add=0.3, damping=0.0001,
                            hess_folder='./cache/',
                            criterion=torch.nn.MSELoss(reduction='sum'),
                            l2_reg=0.0001)
labels[added_keys] = 1.0
new_dataset = KeyDataset(labels)
updated_model.plot(new_dataset, "Updated L2 Model")

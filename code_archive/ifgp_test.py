from models.binary_logistic_nn import BinaryLogisticNN
from utils.mfac_static import HInvFastSwap
import torch
from utils.key_dataset import KeyDataset
from utils.single_step_updates import update_params_, compute_loss_gradient_, ihgp_update
from torch.utils.data import DataLoader, Subset
import numpy as np
import copy
import os
import random

n_keys = 2000
pos_keys = set(range(20)) | set(range(500, 560)) | set(range(900, 1200)) | \
        set(range(1800, 2000))
random_keys = set(random.choices(range(2000), k=100))
pos_keys = pos_keys ^ random_keys
pos_keys = np.asarray(list(pos_keys)).astype(int)

added_keys = set(range(100, 400))
added_keys = np.asarray(list(added_keys))
np.random.shuffle(added_keys)
added_keys = np.sort(added_keys[:100:])


# Initialize labels as 0 + some noise
labels = np.random.choice([0.0, 0.15, 0.25], n_keys,
                          p=[0.90, 0.05, 0.05])
# labels = np.random.choice([0.0, 0.2], n_keys, p=[0.6, 0.4])
labels[pos_keys] = 1.0
dataset = KeyDataset(labels)

# Train a model on the initial data and plot the result
model = BinaryLogisticNN(hidden_size=200,
                         mean=dataset.mean, std=dataset.std)
title = "Fisher"
if not os.path.exists("./cache/" + title + "_model.pt"):
    model.do_train(dataset, batch_size=2000, num_epochs=50000,
                   lr=0.11, step_size=3000, decay=0.95, noise=0.02,
                   title="Initial Model")
    model.save(title + "_model.pt")
else:
    model.load(title + "_model.pt")
model.plot(dataset, "Initial Model")

new_dataset = dataset.changed_labels(added_keys,
                                     np.ones(added_keys.size))
new_loader = DataLoader(new_dataset, batch_size=n_keys)

print("computing grads")
# Get gradients w.r.t new loss
device = "cuda"
criterion = torch.nn.BCEWithLogitsLoss(reduction='none')
model.eval()
model.zero_grad()

criterion = criterion.to(device)

m, d = n_keys, model.params().size()[0]

grads = torch.zeros(m, d).to(device)

eps_add = 0.2
damp = 0.00001

for i, (sample, target) in enumerate(new_loader):
    sample, target = sample.to(device), target.to(device)
    output = model(sample).view(-1)

    loss = criterion(output, target)
    for j in range(m):
        model.zero_grad()
        loss[j].backward(retain_graph=True)
        grads[j] += torch.cat([param.grad.view(-1)
                               for param in model.parameters()]).view(-1)

grads[added_keys, :] *= eps_add

full_grad = False

if full_grad:
    grad = (torch.sum(grads, dim=0) / m).cpu().detach()
else:
    added_dataset = Subset(new_dataset, added_keys)
    added_loader = DataLoader(added_dataset, batch_size=100)

    removed_dataset = Subset(dataset, added_keys)
    removed_loader = DataLoader(removed_dataset, batch_size=100)

    added_grad = compute_loss_gradient_(model,
                                        added_loader, 'cuda')
    removed_grad = compute_loss_gradient_(model,
                                          removed_loader, 'cuda')
    grad = (eps_add*added_grad - removed_grad) / n_keys
    grad = grad.cpu().detach()

grads = grads.cpu().detach()

print("applying update")
# Compute inverse fisher gradient product
inv_fisher = HInvFastSwap(grads, damp=damp, npages=1, cpu='cpu',
                          gpu=torch.device('cuda'))

ifgp = inv_fisher.mul(grad)

# Apply update
fisher_model = copy.deepcopy(model)
update_params_(fisher_model, -ifgp, device)
fisher_model.plot(new_dataset, title + " Updated Model")

# Apply Hessian update
hessian_model = copy.deepcopy(model)
hessian_model = ihgp_update(hessian_model, dataset, added_keys,
                            added_keys,
                            changed_labels=np.ones(added_keys.size),
                            device="cuda", eps_add=eps_add,
                            damping=damp, hess_folder='./cache/b/',
                            full_grad=False)
hessian_model.plot(new_dataset, "Hessian Updated Model")

# Apply Hessian update
hessian_model2 = copy.deepcopy(model)
hessian_model2 = ihgp_update(hessian_model2, dataset, added_keys,
                             added_keys,
                             changed_labels=np.ones(added_keys.size),
                             device="cuda", eps_add=eps_add,
                             damping=damp, hess_folder='./cache/a/',
                             full_grad=True)
hessian_model2.plot(new_dataset, "Hessian Updated Model full grad")

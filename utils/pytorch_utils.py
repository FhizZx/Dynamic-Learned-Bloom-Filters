''' Utility functions that make working with pytorch nn's more convenient
'''

from utils.hessian_utils import hessian_vector_product
import torch
import os
from torch.utils.data import DataLoader
from torch.nn import BCEWithLogitsLoss, MSELoss
import numpy as np


def params(model):
    """ Returns 1D tensor with all the parameters (weights/biases) of the model
    """
    return torch.cat([torch.flatten(param)
                      for param in model.parameters()
                      if param.requires_grad])


def num_params(model):
    """ Number of trainable parameters of the model"""
    return params(model).size()[0]


def compute_loss_gradients_(model, dataset, device, weights=None,
                            reduction='none', l2=False, l2_reg=0.0):
    """ Compute the gradients of the loss function at the data samples in the
    dataset.

    Parameters:
        model - a pytorch nn
        dataset - a binary labelled dataset
        weights - a vector of scalings for each sample in the dataset
        reduction - none / mean / sum
        l2 - whether the MSE loss criterion is used
        l2_reg - the regularization term for l2 loss
    """
    model.eval()
    model.zero_grad()

    if not l2:
        criterion = BCEWithLogitsLoss(reduction='none')
    else:
        criterion = MSELoss(reduction='none')
    criterion.to(device)

    # Compute loss
    x, target = dataset[:]
    x, target = x.to(device), target.to(device)
    y = model(x).view(-1)
    loss = criterion(y, target)

    # Compute loss gradients for each data sample
    m = x.shape[0]
    d = num_params(model)
    grads = torch.zeros((m, d)).to(device)
    for j in range(m):
        model.zero_grad()
        loss[j].backward(retain_graph=True)
        grads[j] = torch.cat([param.grad.view(-1)
                             for param in model.parameters()]).view(-1)

    # Apply sample weighting
    if weights is not None:
        grads *= weights.view(-1, 1).to(device)

    # Apply reduction
    if reduction == 'mean':
        grads = (torch.sum(grads, dim=0) / m)
    elif reduction == 'sum':
        grads = (torch.sum(grads, dim=0))

    # Add l2 regularization
    if l2_reg > 0 and reduction != 'none':
        reg = 0
        for param in model.parameters():
            reg += param.sum()
        grads += l2_reg * reg

    return grads


def compute_loss_hessian_(model, dataset, device,
                          reduction='sum', l2=False, l2_reg=0.0,
                          cache_path=None):
    """ Compute the Hessian of the loss at the data samples in the dataset.

        Parameters:
            model - some autograd pytorch model
            dataset - samples to compute the hessian of the loss on
            reduction - how the sample costs are aggregates in the loss,
                        should be either mean or sum
            l2 - whether to use MSELoss as a criterion
            l2_reg - regularization parameter if l2 loss is used
            cache_path - where to store/load the hessian

        credit to Alexandra for hessian vector product. """

    if not l2:
        criterion = BCEWithLogitsLoss(reduction=reduction)
    else:
        criterion = MSELoss(reduction=reduction)
    criterion.to(device)

    loader = DataLoader(dataset, batch_size=1000)

    if cache_path and os.path.exists(cache_path):
        hess = torch.load(cache_path)
    else:
        hess = None
        d = num_params(model)
        for idx in range(d):
            e = np.zeros(d)
            e[idx] = 1.0
            vector = torch.from_numpy(e).float().to(device)
            hess_col = hessian_vector_product(vector, loader, model, device,
                                              l2_reg=l2_reg,
                                              criterion=criterion)
            hess_col = hess_col.detach().cpu().numpy().reshape(-1, 1)
            if hess is None:
                hess = hess_col
            else:
                hess = np.hstack([hess, hess_col])
        if cache_path:
            torch.save(hess, cache_path)
    return torch.Tensor(hess).to(device)


def update_params_(model, vector, device):
    """ Sum up the model's trainable parameters with vector """
    start_idx = 0
    params = [p for p in model.parameters() if p.requires_grad]
    for param in params:
        end_idx = start_idx + param.numel()
        param.data.add_(vector[start_idx:end_idx].view(param.data.shape))
        start_idx = end_idx
    return model

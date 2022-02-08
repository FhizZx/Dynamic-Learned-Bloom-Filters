import torch
from torch.nn import BCEWithLogitsLoss


def hessian_vector_product(vector, loader, model, device, l2_reg=0.0,
                           criterion=BCEWithLogitsLoss(reduction='sum')):
    vector = vector.to(device)
    param_list = [param for param in model.parameters() if param.requires_grad]
    vector_list = []
    offset = 0
    for param in param_list:
        vector_list.append(
            vector[offset:offset + param.numel()].detach().view_as(param).to(param.device))
        offset += param.numel()
    model.eval()
    model.zero_grad()
    for input, target in loader:
        input, target = input.to(device), target.to(device)
        output = model(input).float().view(-1)
        loss = criterion(output, target)
        grad_list = torch.autograd.grad(loss, param_list, create_graph=True)
        dL_dvec = torch.zeros(1)
        dL_dvec = dL_dvec.to(device)
        for v, g in zip(vector_list, grad_list):
            dL_dvec += torch.sum(v * g)
        dL_dvec.backward()
    model.eval()
    hv = torch.cat([param.grad.view(-1) for param in param_list]).view(-1)
    if l2_reg > 0:
        hv = hv + l2_reg * vector
    return hv

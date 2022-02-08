import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from torch.autograd import Variable
from torch.utils.data import DataLoader

device = ("cuda" if torch.cuda.is_available() else "cpu")


def np_to_pt(x):
    x = torch.from_numpy(x.squeeze()).type(
        torch.FloatTensor).view(-1, 1)
    if device == "cuda":
        x = Variable(x).cuda()
    return x


class BinaryLogisticNN(nn.Module):
    def __init__(self, shape, mean, std, relu_activation=False):
        super(BinaryLogisticNN, self).__init__()
        # Single input, single sigmoid output with one hidden layers
        self.shape = shape
        layer = nn.Linear(1, shape[0])
        if relu_activation:
            nn.init.kaiming_uniform_(layer.weight.data, a=0.01)
            nn.init.normal_(layer.bias.data, 0.0, 1.0)
            modules = [layer, nn.LeakyReLU()]
        else:
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.normal_(layer.bias.data, 0.0, 1.0)
            modules = [layer, nn.Sigmoid()]

        for i in range(len(shape) - 1):
            layer = nn.Linear(shape[i], shape[i + 1])
            nn.init.xavier_uniform_(layer.weight.data)
            nn.init.normal_(layer.bias.data, 0.0, 1.0)
            modules += [layer, nn.Sigmoid()]
        layer = nn.Linear(shape[-1], 1)
        nn.init.xavier_uniform_(layer.weight.data)
        nn.init.normal_(layer.bias.data, 0.0, 1.0)
        modules += [layer]
        self.net = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()
        self.criterion = torch.nn.BCEWithLogitsLoss()
        self.mean = mean
        self.std = std
        self.optim_state = None
        self.num_params = torch.cat([torch.flatten(param)
                                    for param in self.parameters()
                                    if param.requires_grad]).size()[0]

        self.to(device)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def predict(self, x):
        self.eval()
        x = np.asarray(x)
        x_tensor = np_to_pt(x)
        out_tensor = self.sigmoid(self(x_tensor))
        return out_tensor.cpu().detach().numpy().squeeze()

    def predict_logits(self, x):
        self.eval()
        x = np.asarray(x)
        x_tensor = np_to_pt(x)
        out_tensor = self(x_tensor)
        return out_tensor.cpu().detach().numpy().squeeze()

    def save(self, name):
        # todo - use relative path
        torch.save(self.state_dict(), "./cache/" + name)

    def load(self, name):
        self.load_state_dict(torch.load("./cache/" + name))

    def params(self):
        return torch.cat([torch.flatten(param)
                         for param in self.parameters()
                         if param.requires_grad])

    def reset_params(self):
        self = BinaryLogisticNN(self.shape, self.mean, self.std)

    def do_train(self, dataset, batch_size, num_epochs, lr,
                 step_size, decay, noise=0.0, title="Model"):
        self.train()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        '''
        if self.optim_state:
            optimizer.load_state_dict(self.optim_state)'''
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=step_size, gamma=decay)

        # Load data for batch training
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        for x, y in dataloader:         # do training on one batch
            x, y = x.to(device), y.to(device)
            for epoch in range(num_epochs):
                noise_t = torch.rand(y.size()).type(torch.FloatTensor) - 0.5
                noise_t = noise_t * noise
                y_noisy = torch.clamp((y + noise_t.to(device)), min=0., max=1.)

                # feed forward
                pred = self(x).view(-1)
                loss = self.criterion(pred, y_noisy)
                # back prop
                optimizer.zero_grad()       # reset the accumulated gradient
                loss.backward()             # compute gradient
                optimizer.step()            # do one descent step
                scheduler.step()            # update learning rate
                if (epoch % step_size == 0):
                    print("epoch: {}, loss: {}, lr: {}".format(
                        epoch, loss.item(), optimizer.param_groups[0]['lr']))
                    self.plot(dataset, title)
        self.optim_state = optimizer.state_dict()

    def plot(self, dataset, title, extra_title="", folder="images/"):
        keys = dataset.keys
        pred = self.predict(keys)
        color = np.full(keys.size, 'r')
        color[dataset.pos_keys] = 'g'

        plt.plot(keys, pred, zorder=1, linewidth=0.3, color="black")
        plt.scatter(keys, pred, zorder=2, c=color, s=6, alpha=1.0)
        plt.title(title + extra_title)
        plt.xlabel("Keys")
        plt.ylabel("Predictor confidence in key being positive")
        green = mpatches.Patch(color='g', label='Positive Keys')
        red = mpatches.Patch(color='r', label='Negative Keys')
        plt.legend(handles=[green, red])
        plt.savefig(folder + title)
        plt.clf()

    def plot_on_axis(self, keys, ax, alpha=1.0, zorder=1, color='black',
                     linewidth=0.3):
        pred = self.predict(np.asarray(keys))
        ax.plot(keys, pred, zorder=zorder, c=color, linewidth=linewidth,
                alpha=alpha)

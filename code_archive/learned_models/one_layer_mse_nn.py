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


# 1-NN used to approximate a function
class OneLayerMSENN(nn.Module):
    def __init__(self, hidden_size, mean, std):
        super(OneLayerMSENN, self).__init__()
        # Single input, single sigmoid output with one hidden layer
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size), nn.Sigmoid(),
            nn.Linear(hidden_size, 1)
        )
        self.criterion = torch.nn.MSELoss(reduction='sum')
        self.mean = mean
        self.std = std
        self.optim_state = None
        self.to(device)

    def forward(self, x):
        return self.net((x - self.mean) / self.std)

    def predict(self, x):
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

    def do_train(self, dataset, batch_size, num_epochs, lr, weight_decay,
                 step_size, decay, noise=0.0, title="Model"):
        self.train()
        optimizer = torch.optim.AdamW(self.parameters(), lr=lr,
                                      weight_decay=weight_decay)
        if self.optim_state:
            optimizer.load_state_dict(self.optim_state)
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
        plt.scatter(keys, pred, zorder=2, c=color, s=4, alpha=0.5)
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

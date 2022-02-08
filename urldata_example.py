""" Example of using an online LBF for the URL dataset"""

from binary_dataset import BinaryDataset
import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from online_learned_models import IFGP_NN_Classifier
from bloom_filters import OnlineLBF
from utils.update_tuning import fpr_count, fnr_count
import os

device = torch.device("cuda")


# Load a binary dataset from file and split into train, test, insertion sets
full_dataset = BinaryDataset.from_csv("./data/urldata_processed.csv",
                                      standardize=True)
# make dataset smaller so that my laptop can handle fisher computations
full_dataset, _ = full_dataset.train_test_split(0.02)
initial_dataset, added_dataset = full_dataset.initial_added_split(0.9, 1)
train_dataset, test_dataset = initial_dataset.train_test_split(0.8)
retrain_dataset, retest_dataset = full_dataset.train_test_split(0.8)

_, num_features = full_dataset.features.shape


# Make neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        layer1 = nn.Linear(num_features, 10)
        nn.init.kaiming_uniform_(layer1.weight.data, a=0.01)
        nn.init.normal_(layer1.bias.data, 0.0, 1.0)
        modules = [layer1, nn.LeakyReLU()]

        layer2 = nn.Linear(10, 10)
        nn.init.xavier_uniform_(layer2.weight.data)
        nn.init.normal_(layer2.bias.data, 0.0, 1.0)
        modules += [layer2, nn.Sigmoid()]

        layer3 = nn.Linear(10, 1)
        nn.init.xavier_uniform_(layer3.weight.data)
        nn.init.normal_(layer3.bias.data, 0.0, 1.0)
        modules += [layer3]

        self.net = nn.Sequential(*modules)
        self.sigmoid = nn.Sigmoid()
        self.to(device)

    def forward(self, x):
        return self.net(x)

    def predict(self, x):
        # x is np array/list
        self.eval()
        x = np.asarray(x)
        x_tensor = torch.from_numpy(x.squeeze()).type(
                         torch.FloatTensor).cuda()
        out_tensor = self.sigmoid(self(x_tensor))
        return out_tensor.cpu().detach().numpy().squeeze()

    def do_train(self, train_dataset, test_datset,
                 num_epochs, batch_size=100, lr=0.1,
                 factor=0.25):
        criterion = torch.nn.BCEWithLogitsLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                               factor=factor,
                                                               patience=2)
        train_loader = DataLoader(train_dataset, batch_size=batch_size,
                                  shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))
        num_train_samples = len(train_dataset)
        num_test_samples = len(test_dataset)

        for epoch in range(num_epochs):
            self.train()
            total_loss = 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                pred = self(x).view(-1)
                loss = criterion(pred, y)
                pred_proba = self.sigmoid(pred)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss * torch.numel(y) / num_train_samples
            scheduler.step(total_loss)
            if optimizer.param_groups[0]['lr'] < 1e-8:  # early stop condition
                break

            if (epoch % 1 == 0):
                self.eval()
                num_correct = 0
                for x, y in test_loader:
                    x, y = x.to(device), y.to(device)
                    pred_proba = self.sigmoid(self(x).view(-1))
                    num_correct += (pred_proba - y < 0.02).float().sum()
                accuracy = 100 * num_correct / num_test_samples
                print("epoch: {}, loss: {}, lr: {}, acc: {}%".format(
                      epoch, total_loss.item(),
                      optimizer.param_groups[0]['lr'],
                      accuracy))
        self.eval()

    def save(self, name):
        torch.save(self.state_dict(), "./cache/" + name)

    def load(self, name):
        self.load_state_dict(torch.load("./cache/" + name))

    def params(self):
        return torch.cat([torch.flatten(param)
                          for param in self.parameters()
                          if param.requires_grad])

    def size(self):
        return 0


num_epochs = 10

# Train initial network
model = Network()
if not os.path.exists("./cache/url_initial_model.pt"):
    model.do_train(train_dataset, test_dataset, num_epochs=num_epochs)
    model.save("url_initial_model.pt")
else:
    model.load("url_initial_model.pt")
online_model = IFGP_NN_Classifier(model, train_dataset)
olbf = OnlineLBF(online_model, train_dataset, fpr_bound=0.05, fnr_bound=0.01)

# Insert extra samples in batches using Fisher update
insertion_batch_size = 20000
batch_datasets = added_dataset.batches_split(insertion_batch_size)

for batch in batch_datasets:
    olbf.insert(batch)

# Some basic metrics about how well the update did
fpr_c = fpr_count(online_model, full_dataset, olbf.threshold)
fnr_c = fnr_count(online_model, full_dataset, olbf.threshold)
print("update fpr_count: {}, update fnr_count: {}".format(fpr_c, fnr_c))

# Retrain with all samples
retrained_model = Network()
if not os.path.exists("./cache/url_retrained_model.pt"):
    retrained_model.do_train(retrain_dataset, test_dataset,
                             num_epochs=num_epochs)
    retrained_model.save("url_retrained_model.pt")
else:
    retrained_model.load("url_retrained_model.pt")
o_retrained_model = IFGP_NN_Classifier(retrained_model, train_dataset)
retrained_olbf = OnlineLBF(o_retrained_model, retrain_dataset,
                           fpr_bound=0.05, fnr_bound=0.01)

fpr_c = fpr_count(o_retrained_model, full_dataset, retrained_olbf.threshold)
fnr_c = fnr_count(o_retrained_model, full_dataset, retrained_olbf.threshold)
print("retrain fpr_count: {}, retrain fnr_count: {}".format(fpr_c, fnr_c))

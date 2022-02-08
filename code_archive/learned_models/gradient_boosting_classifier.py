
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.ensemble import GradientBoostingClassifier


class GradientBoosting():
    def __init__(self, n_estimators, max_leaf_nodes, lr=0.1):
        self.model = GradientBoostingClassifier(n_estimators=n_estimators,
                                                max_leaf_nodes=max_leaf_nodes,
                                                learning_rate=lr)

    def predict(self, x):
        x = np.asarray(x).reshape(-1, 1)
        out = self.model.predict_proba(x)[:, 1]
        return out.squeeze()

    def do_train(self, dataset, title="Model"):
        self.model.fit(dataset.keys.reshape(-1, 1),
                       dataset.labels.round(decimals=1).astype(int))

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

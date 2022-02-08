
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skgarden.mondrian import MondrianForestClassifier


class MondrianForest():
    def __init__(self, n_estimators=10, max_depth=None, min_samples_split=2):
        self.model = MondrianForestClassifier(n_estimators=n_estimators,
                                              max_depth=max_depth,
                                              min_samples_split=min_samples_split)

    def predict(self, x):
        x = np.asarray(x).reshape(-1, 1)
        out = self.model.predict_proba(x)[:, 1]
        return out.squeeze()

    def do_train(self, x, y):
        self.model.partial_fit(x.reshape(-1, 1),
                               y.round(decimals=1).astype(int))

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

import numpy as np
from scipy.stats import entropy
import os
import imageio
import shutil
import matplotlib.pyplot as plt


# False positive rate of the model on the dataset
def fpr(model, dataset, threshold):
    neg_keys, neg_features = dataset.negatives
    neg_output = model.predict(neg_features)
    return 1.0 * (neg_output >= threshold).sum() / neg_keys.size


# Number of false positives of the model on the dataset
def fpr_count(model, dataset, threshold):
    neg_keys, neg_features = dataset.negatives
    neg_output = model.predict(neg_features)
    return (neg_output >= threshold).sum()


# False negative rate of the model on the dataset
def fnr(model, dataset, threshold):
    pos_keys, pos_features = dataset.positives
    pos_output = model.predict(pos_features)
    return 1.0 * (pos_output < threshold).sum() / pos_keys.size


# False negative rate of the model on the dataset
def fnr_count(model, dataset, threshold):
    pos_keys, pos_features = dataset.positives
    pos_output = model.predict(pos_features)
    return (pos_output < threshold).sum()


# Return the KL divergence between the model's output and the target
def divergence_loss(model, dataset):
    labels = dataset.labels + 0.0000001
    output = model.predict(dataset.features) + 0.0000001
    p = labels
    q = output
    return entropy(p, q)


# example of a metric_loss_function
# Return a linear combination of the fnr and fpr of a model as a
# hyper tuning loss
def weighted_error_loss(model, dataset, threshold):
    model_fpr = fpr(model, dataset, threshold)
    model_fnr = fnr(model, dataset, threshold)
    return 0.4*np.exp(1 / (1 - model_fpr)) \
        + 0.6*np.exp(1 / (1 - model_fnr))


# Grid search on the hyper parameters to find the update that minimizes the
# loss of the metric function
def single_step_grid_search_tuning(sse_function, eps_grid, damp_grid,
                                   metric_loss_function, new_dataset,
                                   title="current model",
                                   folder="images/",
                                   threshold=0.5):
    opt_loss = 1000000000.0
    opt_model = None
    gif_dir = folder + '/temp/'
    os.makedirs(gif_dir)
    i = 0
    fpr_m = np.empty((damp_grid.size, eps_grid.size))
    fnr_m = np.empty((damp_grid.size, eps_grid.size))
    for j, damp in np.ndenumerate(damp_grid):
        for k, eps in np.ndenumerate(eps_grid):
            print("eps: {}, damp: {}".format(eps, damp))
            updated_model = sse_function(eps, damp)

            fpr_m[j, k] = fpr_count(updated_model, new_dataset, threshold)
            fnr_m[j, k] = fnr_count(updated_model, new_dataset, threshold)

            i += 1
            loss = metric_loss_function(updated_model)
            print("loss: {}".format(loss))
            if loss < opt_loss:
                opt_loss = loss
                opt_model = updated_model
    # make gif of metrics change
    images = []
    for file_name in sorted(os.listdir(gif_dir)):
        file_path = os.path.join(gif_dir, file_name)
        images.append(imageio.imread(file_path))
    imageio.mimsave(folder+title+'.gif', images, fps=2)
    shutil.rmtree(gif_dir)

    plt.axis('tight')
    plt.axis('off')
    plt.table(cellText=fpr_m, rowLabels=damp_grid,
              colLabels=eps_grid, loc='center')
    plt.savefig(folder + title + ' Classifier FPR Metrics.png')
    plt.rcParams["figure.figsize"] = plt.rcParamsDefault["figure.figsize"]
    plt.clf()

    plot_metrics([(fpr_m, "Classifier FPR"),
                  (fnr_m, "Classifier FNR")],
                 eps_grid, damp_grid, title, folder=folder)

    return opt_model


def plot_metrics(matrices, eps_grid, damp_grid,
                 title, folder="./images/"):
    n = len(matrices)
    fig = plt.figure()
    fig, axes = plt.subplots(n, 1, figsize=(8, 8),
                             sharex=True, sharey=True)
    fig.suptitle("Update metrics", size=16)
    axes[0].set_xticks(range(0, eps_grid.size, 4))
    axes[0].set_xticklabels(eps_grid[::4])
    axes[0].set_yticks(range(damp_grid.size))
    axes[0].set_yticklabels(damp_grid)
    fig.add_subplot(111, frameon=False)
    # hide tick and tick label of the big axis
    plt.tick_params(labelcolor='none', which='both', top=False,
                    bottom=False, left=False, right=False)
    plt.xlabel("Epsilon")
    plt.ylabel("Damping")

    for ax, (mat, name) in zip(axes, matrices):
        a = ax.imshow(mat, cmap='Blues', interpolation='none')
        cbar = fig.colorbar(a, ax=ax)
        cbar.set_label(name, rotation=0, labelpad=50)

    fig.savefig(folder + title + ' Metrics')
    plt.close(fig)
    plt.clf()

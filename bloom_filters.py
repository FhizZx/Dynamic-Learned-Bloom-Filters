""" A collection of Bloom Filters:
        a standard Bloom Filter
        a simple read-only Learned Bloom Filter
        a Learned Bloom Filter that supports updates
"""

from bitstring import BitArray
import mmh3
import math
import numpy as np

THRESHOLD_GRID_SEARCH_SIZE = 25


class BloomFilter:
    """ Standard Bloom Filter

    Attributes:
        a  -- bit array storing the data
        m  -- # bits
        n  -- set capacity - this is a soft cap that ensures the FPR
              (false positive rate) is within the desired error
        k  -- number of hash functions used
        eps - desired false positive rate
    """

    def __init__(self, n, eps):
        self.n = n
        self.eps = eps
        self.k = math.ceil((-1) * math.log(eps) / math.log(2))
        self.m = max(math.ceil(self.n * self.k / math.log(2)),
                     1)
        self.a = BitArray(self.m)
        print("Created Bloom Filter of %d bytes that uses %d hashes" %
              (self.size(), self.k))

    def insert(self, keys):
        indices = set()
        for key in keys:
            indices |= set(map(lambda i: mmh3.hash(str(key), i) % self.m,
                           range(0, self.k)))
        self.a.set(True, indices)

    def __contains__(self, key):
        return all(list(map(
            lambda i: self.a[mmh3.hash(str(key), i) % self.m],
            range(0, self.k))))

    # Size of the Bloom Filter in bytes
    def size(self):
        return self.m / 8


class LearnedBF:
    """ A read-only bloom filter augmented by a learned predictor

        Attributes:
            f -- pre-trained model used as a prefilter to reduce number of
                false positives
            threshold -- used to classify keys based on predictor's outputs
            backup_filter -- bloom filter used to catch all false negatives
    """

    def __init__(self, model, dataset, eps):
        # Set the prefilter to be the given pre-trained model
        self.f = model

        pos_keys, pos_features = dataset.positives
        neg_keys, neg_features = dataset.negatives

        # The predictions of the prefilter on the data
        pos_pred = self.f.predict(pos_features)
        neg_pred = self.f.predict(neg_features)

        # Find the threshold that minimizes memory usage of the backup filter
        self.threshold, bf_fpr, pred_fpr = self.find_opt_threshold_(pos_pred,
                                                                    neg_pred,
                                                                    eps)
        self.pred_fpr = pred_fpr

        # Create a backup bloom filter to hold the false negatives
        # of the prefilter
        false_negative_keys = pos_keys[np.where(pos_pred < self.threshold)]
        self.backup_filter = BloomFilter(false_negative_keys.size, bf_fpr)
        self.backup_filter.insert(false_negative_keys)

        # In practice, the filter wouldn't store the data, but this is
        # for testing purposes
        self.bf_contents = false_negative_keys

    def __contains__(self, elem):
        key, X = elem
        pred = self.f.predict(X).item()
        return pred >= self.threshold or key in self.backup_filter

    @staticmethod
    def find_opt_threshold_(pos_pred, neg_pred, eps):
        """ Perform a grid search on the threshold space to find the one
        which minimizes memory usage between the model and the backup filter
        """
        grid = np.linspace(0.0, 1.0, THRESHOLD_GRID_SEARCH_SIZE)

        opt_threshold = 1.0
        opt_bf_fpr = eps
        opt_bf_n = pos_pred.size
        opt_pred_fpr = 0.0

        for t in grid:
            # empirical false positive rate for the given threshold
            pred_fpr = 1.0 * (neg_pred >= t).sum() / neg_pred.size
            # maximum false positive rate needed for the backup bloom
            # filter to ensure total fpr of at most eps
            if pred_fpr < eps:
                bf_fpr = (eps - pred_fpr) / (1.0000001 - pred_fpr)

                # number of elements needed to be inserted into the backup
                # filter
                bf_n = (pos_pred < t).sum()

                # size of a bloom filter is proportional to n * log(1/fpr)
                if -math.log(bf_fpr) * bf_n <= \
                   -math.log(opt_bf_fpr) * opt_bf_n:
                    opt_bf_fpr = bf_fpr
                    opt_bf_n = bf_n
                    opt_threshold = t
                    opt_pred_fpr = pred_fpr

        return opt_threshold, opt_bf_fpr, opt_pred_fpr

    def size(self):
        return self.f.size() + self.backup_filter.size()


class OnlineLBF(LearnedBF):
    ''' A learned bloom filter that supports inserting new elements '''

    def __init__(self, model, dataset, fpr_bound, fnr_bound):
        LearnedBF.__init__(self, model, dataset, fpr_bound)

    def insert(self, dataset):
        # Update the model with the given dataset
        self.f.online_train(dataset, self.threshold)

        # Here there might be several strategies as to how to go about handling
        # newly appeared false positives / false negatives
        # Probably need to play around with them to see which works best

        pos_keys, pos_features = dataset.positives
        pos_pred = self.f.predict(pos_features)
        new_bf_keys = pos_keys[pos_pred < self.threshold]
        self.backup_filter.insert(new_bf_keys)


'''
    # Assess how accurate the single step insertion update was
    def insertion_metrics_(self, updated_model, new_dataset, added_elems):

        # Measure false positive rate of the predictor after the update
        # High value => false positive rate increase
        updated_neg_keys = new_dataset.neg_keys
        updated_neg_output = updated_model.predict(updated_neg_keys)
        filter_fpr = 1.0 * (updated_neg_output >= self.threshold).sum() \
            / updated_neg_keys.size

        # Measure the FNR of the LBF on old positive keys,
        # i.e. proportion of old positive keys unaccounted for by both
        # the prefilter model and the backup bloom filter
        # High value => false negative rate increase
        old_pos_keys = np.setdiff1d(self.dataset.pos_keys, self.bf_contents)

        initial_output = self.f.predict(self.dataset.keys)
        initial_true_pos = np.asarray([x for x in old_pos_keys
                                      if initial_output[x] >= self.threshold])
        updated_old_true_pos_output = updated_model.predict(initial_true_pos)
        lbf_fnr = (updated_old_true_pos_output < self.threshold).sum()
        lbf_fnr *= 1.0 / old_pos_keys.size

        # Measure the FNR of newly added keys, i.e. proportion of added keys
        # whose prediction is still not above the threshold after the update
        # High value => Higher memory usage
        updated_new_pos_keys_output = updated_model.predict(added_elems)
        filter_fnr = \
            (updated_new_pos_keys_output < self.threshold).sum()*1.0 \
            / added_elems.size

        # Measure total KL divergence between the desired updated model
        # (old model + 1.0 labels at the insertion points) and the single step
        # updated model
        updated_labels = new_dataset.labels + 0.0000001
        updated_output = updated_model.predict(self.dataset.keys) + 0.0000001
        p = updated_labels
        q = updated_output
        update_divergence = np.tanh(entropy(p, q))

        return filter_fpr, lbf_fnr, filter_fnr, update_divergence

    # A measure of how well the single step update worked for the online lbf
    def insertion_loss_(self, filter_fpr, lbf_fnr, filter_fnr, divergence):
        loss = 0.0
        if filter_fpr > 2 * self.pred_fpr + 0.05 or lbf_fnr > self.fnr:
            loss += 200.0

        loss += 200.0*filter_fpr
        loss += 200.0*lbf_fnr
        loss += 70.0*filter_fnr
        loss += 23.0*divergence
        return loss

    # Exagerate IHGP
    def reinforce_insertion_(self, elems, momentum, span, n_samples):
        # Build up the set of elements to sample existing keys from
        sample_set = set()
        for elem in elems:
            sample_set |= set(range(max(0, elem - span),
                                    min(elem + span, len(self.dataset.keys))))
        sample_set = np.setdiff1d(np.asarray(list(sample_set)), elems,
                                  assume_unique=True)
        n_samples = min(n_samples, sample_set.size)
        samples = np.random.choice(sample_set, n_samples, replace=False)
        changed_keys = np.union1d(elems, samples)
        new_dataset = self.dataset.changed_labels(elems, np.ones(elems.size))

        # Exagerate the labels of changed keys
        new_dataset.labels[new_dataset.pos_keys] += momentum
        new_dataset.labels[new_dataset.neg_keys] -= 0.3 * momentum

        changed_labels = new_dataset.labels[changed_keys]

        return changed_keys, changed_labels

    def plot(self, new_model, new_dataset, added_elems,
             title, extra_title="", folder="images/", changed_elems=None):
        keys = self.dataset.keys

        fig = plt.figure()
        fig, [ax1, ax2] = plt.subplots(2, 1, figsize=(8, 8),
                                gridspec_kw={'height_ratios': [0.93, 0.07]})
        ax2.axis('off')
        fig.suptitle(title+extra_title)
        fig.subplots_adjust(hspace=0.06, wspace=0.)

        ax1.set_title('Prefilter Model Predictions')
        ax1.set_ylim([-0.01, 1.01])
        ax1.set_xlim([0, keys.size])
        ax1.set_ylabel('Predictor confidence in key being positive')

        # Plot old and new model predictions
        self.f.plot_on_axis(keys, ax1, alpha=0.8, zorder=2,
                            color='tab:blue', linewidth=0.4)
        new_model.plot_on_axis(keys, ax1, alpha=0.8, zorder=3,
                               color='dimgrey', linewidth=0.8)

        # Plot key distribution
        color = np.empty(keys.size, dtype="object")
        color[keys] = 'g'
        color[new_dataset.neg_keys] = 'r'
        color[self.bf_contents] = 'k'
        pred = new_model.predict(keys)
        old_pos_keys = self.dataset.pos_keys
        false_negatives = old_pos_keys[pred[old_pos_keys] < self.threshold]
        false_negatives = np.setdiff1d(false_negatives, self.bf_contents,
                                       assume_unique=True)
        ax1.scatter(keys, pred, zorder=4, c=color, s=6, alpha=0.4)
        ax1.scatter(added_elems, pred[added_elems], zorder=6,
                    c='lime', s=8, alpha=0.7, marker='P',
                    edgecolor='springgreen',
                    linewidth=0.2)
        ax1.scatter(false_negatives, pred[false_negatives], zorder=7,
                    c='blue', s=4, alpha=0.8, marker="x")
        ax1.scatter(changed_elems, pred[changed_elems], zorder=2,
                    c='silver', s=80, alpha=0.4, linewidth=0.0)
        opp = mpatches.Patch(color='g', label='Positive Keys')
        npp = mpatches.Patch(color='lime', label='Newly Added Keys')
        nnp = mpatches.Patch(color='r', label='Negative Keys')
        bfp = mpatches.Patch(color='k', label='Positive Keys in BF')
        fnp = mpatches.Patch(color='blue', label='False Negatives')

        # Plot threshold line
        ax1.axhline(y=self.threshold, color='gold', zorder=1, lw=3.4,
                    label='Threshold', alpha=0.8)
        fig.legend(handles=[opp, npp, nnp, bfp, fnp], loc="lower left",
        bbox_to_anchor=(0.003, 0.003))

        fig.savefig(folder + title)
        plt.close(fig)
        plt.clf()

    def plot_metrics(self, matrices, title, folder="images/"):
        n = len(matrices)
        fig = plt.figure()
        fig, axes = plt.subplots(n, 1, figsize=(8, 8),
                                 sharex=True, sharey=True)
        fig.suptitle("Update metrics", size=16)
        axes[0].set_xticks(range(0, EPS_SEARCH_GRID.size, 4))
        axes[0].set_xticklabels(EPS_SEARCH_GRID[::4])
        axes[0].set_yticks(range(DAMP_SEARCH_GRID.size))
        axes[0].set_yticklabels(DAMP_SEARCH_GRID)
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



        # Plot matrices
        self.plot_metrics([(filter_fpr_m, "Filter FPR \n (Total FPR)"),
                           (lbf_fnr_m,
                           "LBF FNR \n (Total FNR)"),
                           (filter_fnr_m,
                            "Filter FNR \n (Memory usage)"),
                           (divergence_m,
                            "Update Divergence")],
                          title, folder=dir+'metrics/')
'''

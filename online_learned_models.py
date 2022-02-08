""" Collection of learned models used as prefilters for the learned bloom
filter.
"""
import numpy as np
import torch
from utils.mfac_static import HInvFastSwap
from utils.pytorch_utils import compute_loss_gradients_, update_params_


class BinaryClassifier:
    def predict(self, X):
        """ Returns the confidence of the model in each element of X being
        positive (labelled as 1), as a value between 0 and 1. """
        pass

    def size(self):
        """ Size of the model in bytes """
        pass


class OnlineBinaryClassifier(BinaryClassifier):
    def online_train(self, dataset, threshold):
        pass


class IFGP_NN_Classifier(OnlineBinaryClassifier):
    """ An online binary classifier using a neural network and the inverse
    Fisher gradient product update.
    """

    def __init__(self, nn_model, initial_dataset):
        self.nn_model = nn_model

        ''' You don't need to store the whole dataset because you can
        incrementally update the Inverse Fisher Matrix with newly added
        datapoints using the Sherman Morrison lemma.

        So ideally what you would do is, after training the initial net, you
        find and store the inverse Fisher on the initial dataset. Then, for
        each batch of newly added samples you would somehow find the optimal
        hyper parameters for the Fisher update (maybe even apply it several
        times or in combination with some simple gradient descent), and then
        update the stored fisher matrix (without needing access to the
        previously used data points, just the fisher matrix should be enough)
        '''
        self.dataset = initial_dataset
        grads = compute_loss_gradients_(self.nn_model, self.dataset,
                                        device="cuda", weights=None,
                                        reduction='none').cpu().detach()
        damping = 1.0

        # If the model is large enough, consider using block fisher instead
        self.inv_fisher = HInvFastSwap(grads, damp=damping, npages=1, cpu='cpu',
                                       gpu=torch.device('cuda'))

    def predict(self, X):
        return self.nn_model.predict(X)

    def online_train(self, added_dataset, threshold):
        """ Apply the IHGP update

            There's many approaches to explore here in order to obtain an
        optimal update. You need to find the right hyperparameters for the
        scaling and the damping of the update. Also need to decide on which
        metrics you care about the most.

        One approach could be to try several values of eps/damp and see which
        gives the updated model with the fewest false positives/ false negs
        """

        # Update dataset with newly inserted samples
        # But again, you can be smart about the update and do it without
        # requiring knowledge of past data samples, so don't need to compute
        # the grads and the inv fisher fully every time, it's enough to do it
        # for the new samples and then apply sherman morrison lemma

        # But as you can see I'm not doing that here
        N = len(self.dataset)
        K = len(added_dataset)
        self.dataset = self.dataset.union(added_dataset)

        # For the weights you should compute some scaling hyper parameter
        # epsilon that makes the update optimal. Same for the damping param.
        eps = 1.0
        weights = torch.from_numpy(np.hstack([np.ones(N), np.full(K, eps)]))
        damping = 1.0

        grads = compute_loss_gradients_(self.nn_model, self.dataset,
                                        device="cuda", weights=weights,
                                        reduction='none').cpu().detach()
        self.inv_fisher = HInvFastSwap(grads, damp=damping, npages=1,
                                       cpu='cpu', gpu=torch.device('cuda'))

        # For the gradient used in the IFGP, can assume the loss is minimal,
        # (and so the gradient is 0) at old data points, so enough to just
        # compute the gradient for new datapoints
        grad_weights = torch.from_numpy(np.full(K, eps / (N + K)))
        grad = compute_loss_gradients_(self.nn_model, added_dataset,
                                       device="cuda", weights=grad_weights,
                                       reduction='mean')
        ifgp = self.inv_fisher.mul(grad)
        self.nn_model = update_params_(self.nn_model, -ifgp, device="cuda")

    def size(self):
        return self.nn_model.size()

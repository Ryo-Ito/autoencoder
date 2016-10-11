import tensorflow as tf
from autoencoder import Autoencoder


class SparseAutoencoder(Autoencoder):
    """autoencoder to learn overcomplete features"""
    def __init__(self, inputs, n_input, n_hidden, sparsity):
        self.inputs = inputs
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.sparsity = sparsity

        self._forward()

    def _reconst_error(self):
        return tf.nn.l2_loss(self.inputs - self.outputs) / tf.shape(self.inputs)[0]

    def _kl_divergence(self):
        active_rates = tf.reduce_mean(self.h, 0)
        return tf.reduce_sum(
            self.sparsity * tf.log(self.sparsity / active_rates)
            + (1 - self.sparsity) * tf.log((1 - self.sparsity) / (1 - active_rates)))

    def cost(self):
        return self._reconst_error() + self._kl_divergence()

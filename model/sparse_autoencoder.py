import tensorflow as tf
from autoencoder import Autoencoder


class SparseAutoencoder(Autoencoder):
    """autoencoder to learn sparse features"""
    def __init__(self, inputs, n_input, n_hidden):
        super(SparseAutoencoder, self).__init__(inputs, n_input, n_hidden)

    def kl_divergence(self, sparsity):
        active_rates = tf.reduce_mean(self.h, 0)
        return tf.reduce_sum(
            sparsity * tf.log(sparsity / active_rates)
            + (1 - sparsity) * tf.log((1 - sparsity) / (1 - active_rates)))

    def weight_decay(self):
        return tf.nn.l2_loss(self.fc1w) + tf.nn.l2_loss(self.fc2w)

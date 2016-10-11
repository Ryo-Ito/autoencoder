import tensorflow as tf


class Autoencoder(object):
    """autoencoder to encode images"""
    def __init__(self, inputs, n_input, n_hidden):
        self.inputs = inputs
        self.n_input = n_input
        self.n_hidden = n_hidden

        self._forward()

    def _forward(self):
        self.fc1w = tf.Variable(
            tf.truncated_normal(
                [self.n_input, self.n_hidden],
                dtype=tf.float32,
                stddev=0.1,
                name="weights"))
        self.fc1b = tf.Variable(
            tf.constant(
                value=0.,
                dtype=tf.float32,
                shape=[self.n_hidden]))
        h = tf.nn.bias_add(tf.matmul(self.inputs, self.fc1w), self.fc1b)
        self.h = tf.nn.sigmoid(h)

        self.fc2w = tf.Variable(
            tf.truncated_normal(
                shape=[self.n_hidden, self.n_input],
                dtype=tf.float32,
                stddev=0.1,
                name="weights"))
        self.fc2b = tf.Variable(
            tf.constant(
                value=0.,
                dtype=tf.float32,
                shape=[self.n_input]))
        self.outputs = tf.nn.bias_add(tf.matmul(self.h, self.fc2w), self.fc2b)

    def cost(self):
        return tf.nn.l2_loss(self.inputs - self.outputs) / tf.to_float(tf.shape(self.inputs)[0])

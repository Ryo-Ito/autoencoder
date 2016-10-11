import matplotlib.pyplot as plt
import tensorflow as tf
import input_data


N_HIDDEN = 400
ITER_MAX = 100000
BATCH_SIZE = 50
DISPLAY_STEP = 100


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
        self.output = tf.nn.bias_add(tf.matmul(self.h, self.fc2w), self.fc2b)


def main():
    mnist = input_data.read_data_sets("MNIST_data/")
    X_test = mnist.test.images[:32]
    print X_test.shape

    inputs = tf.placeholder(tf.float32, [None, 28 * 28])
    autoencoder = Autoencoder(inputs, 28 * 28, N_HIDDEN)
    cost = tf.reduce_mean(tf.reduce_sum(tf.squared_difference(inputs, autoencoder.output), 1), 0)
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(ITER_MAX):
            batch = mnist.train.next_batch(BATCH_SIZE)
            loss, _ = sess.run((cost, optimizer), feed_dict={inputs: batch[0]})
            if i % DISPLAY_STEP == 0:
                print "step %5d, cost %f" % (i, loss)

        output = sess.run(autoencoder.output, feed_dict={inputs: X_test})

    plt.figure()
    for i in xrange(len(output)):
        img = output[i, :]
        plt.subplot(8, 8, 2 * i + 1)
        plt.imshow(img.reshape(28, 28), cmap='gray')
        plt.axis('off')
        x = X_test[i, :]
        plt.subplot(8, 8, 2 * i + 2)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

import tensorflow as tf
from model import SparseAutoencoder
import input_data


N_HIDDEN = 2000
SPARSITY = 0.001
ITER_MAX = 100000
BATCH_SIZE = 50
DISPLAY_STEP = 1000


def main():
    mnist = input_data.read_data_sets("MNIST_data/")

    inputs = tf.placeholder(tf.float32, [None, 28 * 28])
    encoder = SparseAutoencoder(inputs, n_input=28 * 28, n_hidden=N_HIDDEN, sparsity=SPARSITY)
    cost = encoder.cost()
    optimizer = tf.train.AdamOptimizer().minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(ITER_MAX):
            batch = mnist.train.next_batch(BATCH_SIZE)[0]
            loss, _ = sess.run((cost, optimizer), feed_dict={inputs: batch})
            if i % DISPLAY_STEP == 0:
                print "step %5d, cost %f" % (i, loss)
        saver.save(sess, "sparse_autoencoder.ckpt")


if __name__ == '__main__':
    main()

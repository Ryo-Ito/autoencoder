import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from model import SparseAutoencoder


def main():
    parser = argparse.ArgumentParser(description="train sparse autoencoder")
    parser.add_argument('--data_dir', type=str, default="MNIST_data/", help='directory of input dataset')
    parser.add_argument('--n_input', type=int, default=28 * 28, help='number of input nodes')
    parser.add_argument('--n_hidden', type=int, default=49, help='number of hidden nodes')
    parser.add_argument('--sparse_coef', type=float, default=1., help='coefficient for sparsity term')
    parser.add_argument('--sparsity', type=float, default=0.1, help='desired sparsity of hidden nodes')
    parser.add_argument('--weight_coef', type=float, default=0.1, help='coefficient for weight decay term')
    parser.add_argument('--iter_max', type=int, default=100000, help='maximum number of iterations')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for gradient-like method')
    parser.add_argument('--batch_size', type=int, default=50, help='size of mini batch')
    parser.add_argument('--display_step', type=int, default=1000, help='how often show learning state')
    args = parser.parse_args()

    mnist = input_data.read_data_sets(args.data_dir)

    inputs = tf.placeholder(tf.float32, [None, args.n_input])
    encoder = SparseAutoencoder(inputs, n_input=args.n_input, n_hidden=args.n_hidden)
    cost = (encoder.reconst_error()
            + args.sparse_coef * encoder.kl_divergence(args.sparsity)
            + args.weight_coef * encoder.weight_decay())
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(args.iter_max):
            batch = mnist.train.next_batch(args.batch_size)[0]
            sess.run(optimizer, feed_dict={inputs: batch})
            if i % args.display_step == 0:
                loss = sess.run(cost, feed_dict={inputs: batch})
                print "step %5d, cost %f" % (i, loss)
        weights = sess.run(encoder.fc1w)

    len_ = int(args.n_hidden ** 0.5)
    assert len_ ** 2 == args.n_hidden
    plt.figure()
    for i in xrange(args.n_hidden):
        plt.subplot(len_, len_, i + 1)
        plt.imshow(weights[:, i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    main()

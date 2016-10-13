import argparse
import tensorflow as tf
from model import SparseAutoencoder
import input_data


def main():
    parser = argparse.ArgumentParser(description="train sparse autoencoder")
    parser.add_argument('--data_dir', type=str, default="MNIST_data/", help='directory of input dataset')
    parser.add_argument('--n_input', type=int, default=28 * 28, help='number of input nodes')
    parser.add_argument('--n_hidden', type=int, default=400, help='number of hidden nodes')
    parser.add_argument('--sparse_coef', type=float, default=0.1, help='coefficient for sparsity term')
    parser.add_argument('--sparsity', type=float, default=0.001, help='desired sparsity of hidden nodes')
    parser.add_argument('--weight_coef', type=float, default=0.001, help='coefficient for weight decay term')
    parser.add_argument('--iter_max', type=int, default=100000, help='maximum number of iterations')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate for gradient-like method')
    parser.add_argument('--batch_size', type=int, default=50, help='size of mini batch')
    parser.add_argument('--display_step', type=int, default=1000, help='how often show learning state')
    parser.add_argument('--output', type=float, default="sparse_autoencoder.ckpt", help="file name of trained model")
    args = parser.parse_args()

    mnist = input_data.read_data_sets(args.data_dir)

    inputs = tf.placeholder(tf.float32, [None, args.n_input])
    encoder = SparseAutoencoder(inputs, n_input=args.n_input, n_hidden=args.n_hidden)
    cost = (encoder.reconst_error()
            + args.sparse_coef * encoder.kl_divergence(args.sparsity)
            + args.weight_coef * encoder.weight_decay())
    optimizer = tf.train.AdamOptimizer(args.learning_rate).minimize(cost)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in xrange(args.iter_max):
            batch = mnist.train.next_batch(args.batch_size)[0]
            loss, _ = sess.run((cost, optimizer), feed_dict={inputs: batch})
            if i % args.display_step == 0:
                print "step %5d, cost %f" % (i, loss)
        saver.save(sess, args.output)


if __name__ == '__main__':
    main()

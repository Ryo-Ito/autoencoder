import argparse
import matplotlib.pyplot as plt
import tensorflow as tf
from model import SparseAutoencoder


def main():
    parser = argparse.ArgumentParser(description="train sparse autoencoder")
    parser.add_argument('--model', type=str, default="sparse_autoencoder.ckpt", help='filename of the model to visualize')
    parser.add_argument('--n_input', type=int, default=28 * 28, help='number of input nodes')
    parser.add_argument('--n_hidden', type=int, default=400, help='number of hidden nodes')
    parser.add_argument('--output', type=str, default="", help="filename to save learned filters")
    args = parser.parse_args()

    inputs = tf.placeholder(tf.float32, [None, args.n_input])
    encoder = SparseAutoencoder(inputs, args.n_input, args.n_hidden)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, args.model)
        weights = sess.run(encoder.fc1w)

    len_ = int(args.n_hidden ** 0.5)
    plt.figure()
    for i in xrange(args.n_hidden):
        plt.subplot(len_, len_, i + 1)
        plt.imshow(weights[:, i].reshape(28, 28), cmap='gray')
        plt.axis('off')
    plt.save(args.output)

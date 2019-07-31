import tensorflow as tf
import numpy as np
from argparse import ArgumentParser
from utils import *

# Sets the threshold for what messages will be logged.
tf.logging.set_verbosity(tf.logging.ERROR)

# Set the number of evaluation
BATCH_SIZE = 1
# Fix random state
RANDOM_STATE = 42

def preprocess_test():
    # Data Preparation
    # ==================================================

    # Load data
    info("Loading data...")
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # Training dataset is not utilized in training process
    # X_train = X_train.reshape(X_train.shape[0], -1)
    # y_train = y_train.astype('int32')
    # X_train = X_train.astype('float32') / 255.

    X_test = X_test.reshape(X_test.shape[0], -1)
    # Apply scaling of input data
    X_test = X_test.astype('float32') / 255.
    y_test = y_test.astype('int32')

    info("Number of test data: {:d}".format(X_test.shape[0]))
    debug("X Test dtype/shape: {}, {}".format(X_test.dtype, X_test.shape))
    debug("y Test dtype/shape: {}, {}".format(y_test.dtype, y_test.shape))
    return X_test, y_test

def restore(checkpoint, X_test, y_test, batch_size=1):
    # Model Restoration
    # ==================================================
    reset_graph()

    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('{}.meta'.format(checkpoint))
        saver.restore(sess, checkpoint)
        tf.get_default_graph()

        X = sess.graph.get_tensor_by_name("X:0")
        logits = sess.graph.get_operation_by_name("dnn/bn_logit/batchnorm/add_1").outputs[0]
        for i in range(batch_size):
            Z = sess.run(logits, feed_dict={X: X_test[i,:].reshape(1,-1)})
            y_pred = np.argmax(Z, axis=1)
            info("Test index: {0}, Predicted class: {1}, Actual class: {2}"
                 .format(i, y_pred[0], y_test[i]))

        # Check the name of variables or operations in restored graph
        # for op in tf.get_default_graph().get_operations():
        #     debug(op.name)
        # all_vars = tf.global_variables()
        # for v in all_vars:
        #     debug(v.name)

def build_parser():
    parser = ArgumentParser()
    parser.add_argument('-ckpt', '--checkpoint', type=str, dest='checkpoint',
                        help='.ckpt file to load checkpoint from (required)', required=True)

    # parser.add_argument('--device', type=str,
    #                     dest='device',help='device to perform compute on',
    #                     metavar='DEVICE', default=DEVICE)

    parser.add_argument('-bs', '--batch-size', type=int, dest='batch_size',
                        help='batch size for feed forwarding (default: %(default)s)', default=BATCH_SIZE)

    # else
    parser.add_argument('-out', '--output_dir', type=str, dest='output_dir',
                        help='log directory to save (default: %(default)s)', default='outputs')
    parser.add_argument('-level', '--log_level', choices=['debug', 'info', 'warning', 'error'],
                        help='(default: %(default)s)', default='debug')

    return parser

def check_opts_test(opts):
    exists(opts.checkpoint + '{}'.format('.meta'), 'Checkpoint not found!')
    assert opts.batch_size > 0

def main():
    parser = build_parser()
    opts = parser.parse_args()
    check_opts_test(opts)
    setup_log_test(opts)

    X_test, y_test = preprocess_test()

    args = [opts.checkpoint,
            X_test, y_test]

    kwargs = {
        "batch_size": opts.batch_size
    }

    restore(*args, **kwargs)

if __name__ == '__main__':
    main()
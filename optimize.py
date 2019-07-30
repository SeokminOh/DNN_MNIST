import tensorflow as tf
import numpy as np
import types
from utils import *
from model import DNN

# A bit hacky, to remove warning TF2.0 message
if type(tf.contrib) != types.ModuleType:
    tf.contrib._warning = None
# Sets the threshold for what messages will be logged.
tf.logging.set_verbosity(tf.logging.ERROR)

# Size of input data (MNIST)
HEIGHT, WIDTH = 28, 28
# Fix random state
RANDOM_STATE = 42

# to make this notebook's output stable across runs
def reset_graph(seed=RANDOM_STATE):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)

def preprocess(train_ratio=0.9):
    # Data Preparation
    # ==================================================

    # Load data
    info("Loading data...")
    mnist = tf.keras.datasets.mnist
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    X_train = X_train.reshape(X_train.shape[0], -1)
    y_train = y_train.astype('int32')
    # Apply scaling of input data
    X_train = X_train.astype('float32') / 255.

    # Test dataset is not utilized in training process
    #X_test = X_test.reshape(X_test.shape[0], -1)
    #X_test = X_test.astype('float32') / 255.
    #y_test = y_test.astype('int32')

    train_val_index = round(X_train.shape[0]*train_ratio)
    X_train, X_val = X_train[:train_val_index, :], X_train[train_val_index:, :]
    y_train, y_val = y_train[:train_val_index], y_train[train_val_index:]

    info("Number of data: {:d}".format(X_train.shape[0]+X_val.shape[0]))
    info("Train/val split: {:d}/{:d}".format(X_train.shape[0], X_val.shape[0]))
    debug("X Train/val dtype/shape: {}, {}, {}, {}".format(X_train.dtype, X_train.shape, X_val.dtype, X_val.shape))
    debug("y Train/val dtype/shape: {}, {}, {}, {}".format(y_train.dtype, y_train.shape, y_val.dtype, y_val.shape))
    return X_train, y_train, X_val, y_val

def optimize(X_train, y_train, X_val, y_val,
             n_hidden1=300, n_hidden2=100, dropout_rate=0.5,
             epochs=10, batch_size=64, learning_rate=1e-3):
    # Training
    # ==================================================

    # Generate batches
    def fetch_batch(batch_size, iteration, epoch):
        np.random.seed(epoch)
        shuffled_indices = np.random.permutation(X_train.shape[0])
        indices = shuffled_indices[batch_size * iteration: batch_size * (iteration + 1)]
        return X_train[indices, :], y_train[indices]

    reset_graph()

    dnn = DNN(
        height=HEIGHT,
        width=WIDTH,
        n_outputs=10,
        n_hidden1=n_hidden1,
        n_hidden2=n_hidden2,
        dropout_rate=dropout_rate,
        seed=RANDOM_STATE)

    n_train = X_train.shape[0]

    # Define Training procedure
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(dnn.loss)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", get_time()))
    info("Writing to {}\n".format(out_dir))

    # Summaries for loss
    loss_summary = tf.summary.scalar("loss", dnn.loss)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, tf.get_default_graph())

    # Val summaries
    val_summary_op = tf.summary.merge([loss_summary])
    val_summary_dir = os.path.join(out_dir, "summaries", "val")
    val_summary_writer = tf.summary.FileWriter(val_summary_dir, tf.get_default_graph())

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(max_to_keep=10)

    session_conf = tf.ConfigProto(allow_soft_placement=True)
    with tf.Session(config=session_conf) as sess:
        # Initialize all variables
        sess.run(tf.global_variables_initializer())

        for epoch in range(epochs):
            # Training loop. For each batch...
            for iteration in range(n_train // batch_size):
                X_batch, y_batch = fetch_batch(batch_size, iteration, epoch)
                sess.run([train_op, extra_update_ops],
                         feed_dict={dnn.training: True, dnn.X: X_batch, dnn.y: y_batch})

            # Evaluates model on a training set
            summary_train, acc_train = sess.run([train_summary_op, dnn.accuracy],
                                                feed_dict={dnn.X: X_batch, dnn.y: y_batch})
            train_summary_writer.add_summary(summary_train, global_step=epoch)
            # Evaluates model on a validation set
            summary_val, acc_val = sess.run([val_summary_op, dnn.accuracy],
                                                feed_dict={dnn.X: X_val, dnn.y: y_val})
            val_summary_writer.add_summary(summary_val, global_step=epoch)
            info("Epoch: {0:3d}, Train accuracy: {1:.4f}, Val accuracy: {2:.4f}"
                 .format(epoch, acc_train, acc_val))

            # Save model
            path = saver.save(sess, checkpoint_prefix, global_step=epoch)
            debug("Saved model checkpoint to {}".format(path))

        cmd_text = 'python evaluate.py --checkpoint %s ...' % checkpoint_prefix
        info("Training complete. For evaluation: {}".format(cmd_text))
import tensorflow as tf

class DNN(object):
    """
    A DNN for digit classification.
    Uses an hidden layer, followed by a batch normalization, activation and dropout layer.
    """
    def __init__(self, height, width, n_outputs, n_hidden1, n_hidden2, dropout_rate=0.1, seed=42):
        n_inputs = height*width
        # Placeholders for input, output and training
        self.X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
        self.y = tf.placeholder(tf.int32, shape=(None), name="y")
        self.training = tf.placeholder_with_default(False, shape=(), name='training')
        self.he_init = tf.contrib.layers.variance_scaling_initializer()
        self.n_hidden1 = n_hidden1
        self.n_hidden2 = n_hidden2
        self.dropout_rate = dropout_rate
        self.n_outputs = n_outputs
        self.seed = seed

        self.logits = self.make_dnn()
        # Calculate mean cross-entropy loss
        with tf.name_scope("loss"):
            self.xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.y, logits=self.logits, name="xentropy")
            self.loss = tf.reduce_mean(self.xentropy, name="loss")
        # Accuracy
        with tf.name_scope("eval"):
            self.correct = tf.nn.in_top_k(predictions=self.logits, targets=self.y, k=1, name="correct")
            self.accuracy = tf.reduce_mean(tf.cast(self.correct, tf.float32), name="accuracy")

    def make_dnn(self):
        with tf.variable_scope("dnn"):
            hidden1 = tf.layers.dense(self.X, self.n_hidden1, kernel_initializer=self.he_init, name="hidden1")
            # Batch normalization
            bn1 = tf.layers.batch_normalization(hidden1, training=self.training, name="bn1")
            # Apply nonlinearity
            elu1 = tf.nn.elu(bn1, name="elu1")
            # Add dropout
            hidden1_drop = tf.layers.dropout(elu1, self.dropout_rate, training=self.training, seed=self.seed, name="drop1")

            hidden2 = tf.layers.dense(hidden1_drop, self.n_hidden2, kernel_initializer=self.he_init, name="hidden2")
            # Batch normalization
            bn2 = tf.layers.batch_normalization(hidden2, training=self.training, name="bn2")
            # Apply nonlinearity
            elu2 = tf.nn.elu(bn2, name="elu2")
            # Add dropout
            hidden2_drop = tf.layers.dropout(elu2, self.dropout_rate, training=self.training, seed=self.seed, name="drop1")

            logits_before_bn = tf.layers.dense(hidden2_drop, self.n_outputs, kernel_initializer=self.he_init, name="outputs")
            # Batch normalization
            logits = tf.layers.batch_normalization(logits_before_bn, training=self.training, name="bn_logit")
        return logits


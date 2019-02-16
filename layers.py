import tensorflow as tf
import numpy as np

# functions
log = lambda x: tf.log(x + 1e-20)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid

# distributions
Normal = tf.distributions.Normal

# layers
dense = tf.layers.dense

# training modules
def cross_entropy(logits, labels):
    return tf.losses.softmax_cross_entropy(logits=logits,
            onehot_labels=labels)

def weight_decay(decay, var_list=None):
    var_list = tf.trainable_variables() if var_list is None else var_list
    return decay*tf.add_n([tf.nn.l2_loss(var) for var in var_list])

def accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

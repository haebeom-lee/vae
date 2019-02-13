import tensorflow as tf
import numpy as np

# functions
exp = tf.exp
log = lambda x: tf.log(x + 1e-20)
logit = lambda x: log(x) - log(1-x)
softplus = tf.nn.softplus
softmax = tf.nn.softmax
relu = tf.nn.relu
sigmoid = tf.nn.sigmoid

# distributions
Normal = tf.distributions.Normal

# layers
dense = tf.layers.dense
flatten = tf.contrib.layers.flatten
dropout = tf.nn.dropout

def conv(x, filters, kernel_size=3, strides=1, **kwargs):
    return tf.layers.conv2d(x, filters, kernel_size, strides,
            data_format='channels_first', **kwargs)

def pool(x, **kwargs):
    return tf.layers.max_pooling2d(x, 2, 2,
            data_format='channels_first', **kwargs)

def global_avg_pool(x):
    return tf.reduce_mean(x, axis=[2, 3])

# training modules
def cross_entropy(logits, labels):
    return tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)

def weight_decay(decay, var_list=None):
    var_list = tf.trainable_variables() if var_list is None else var_list
    return decay*tf.add_n([tf.nn.l2_loss(var) for var in var_list])

def accuracy(logits, labels):
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    return tf.reduce_mean(tf.cast(correct, tf.float32))

def get_train_op(optim, loss, global_step=None, clip=None, var_list=None):
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
        grad_and_vars = optim.compute_gradients(loss, var_list=var_list)
        if clip is not None:
            grad_and_vars = [((None if grad is None \
                    else tf.clip_by_norm(grad, clip)), var) \
                    for grad, var in grad_and_vars]
        train_op = optim.apply_gradients(grad_and_vars, global_step=global_step)
        return train_op

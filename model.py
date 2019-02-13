from layers import *

def encoder(x, zdim, name='encoder', reuse=None):
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)
    x = dense(x, 500, activation=relu, name=name+'/dense2', reuse=reuse)
    mu = dense(x, zdim, name=name+'/mu', reuse=reuse)
    sigma = dense(x, zdim, activation=softplus, name=name+'/sigma', reuse=reuse)
    return mu, sigma

def decoder(x, name='decoder', reuse=None):
    x = dense(x, 500, activation=relu, name=name+'/dense1', reuse=reuse)
    x = dense(x, 500, activation=relu, name=name+'/dense2', reuse=reuse)
    x = dense(x, 784, activation=sigmoid, name=name+'/output', reuse=reuse)
    return x

def autoencoder(x, zdim, training, name='autoencoder', reuse=None):
    mu, sigma = encoder(x, zdim, reuse=reuse)
    z = Normal(mu, sigma).sample() if training else mu
    x_hat = decoder(z, reuse=reuse)

    log_likelihood = tf.reduce_sum(x*log(x_hat) + (1-x)*log(1-x_hat), 1)
    kl = 0.5 * tf.reduce_sum(mu**2 + sigma**2 - log(sigma**2) - 1, 1)
    elbo = tf.reduce_mean(log_likelihood - kl)

    net = {}
    net['elbo'] = elbo
    net['weights'] = tf.trainable_variables()
    return net

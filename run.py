from __future__ import print_function
import tensorflow as tf
import numpy as np
import argparse
import time
import os

from model import autoencoder, decoder
from accumulator import Accumulator
from mnist import mnist_1000

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from scipy.misc import imsave
from scipy.misc import imresize

parser = argparse.ArgumentParser()
parser.add_argument('--batch_size', type=int, default=100)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--savedir', type=str, default=None)
parser.add_argument('--mnist_path', type=str, default='./data')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--zdim', type=int, default=2)
parser.add_argument("--scatter", default=False, action="store_true")
parser.add_argument("--manifold", default=False, action="store_true")
args = parser.parse_args()

os.environ['TF_ENABLE_WINOGRAD_NONFUSED'] = '1'
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_num)
os.environ['CUDA_CACHE_PATH'] = '/st1/hblee/tmp'

savedir = './results/run' \
        if args.savedir is None else args.savedir
if not os.path.isdir(savedir):
    os.makedirs(savedir)

# get data
xtr, ytr, xte, yte = mnist_1000(args.mnist_path)

# placeholders
x = tf.placeholder(tf.float32, [None, 784])
n_train_batches = 1000/args.batch_size
n_test_batches = 1000/args.batch_size

# models
net = autoencoder(x, args.zdim, True) # train
tnet = autoencoder(x, args.zdim, False, reuse=True) # test

# for visualization
z = tf.placeholder(tf.float32, [None, args.zdim])
tdenet = decoder(z, reuse=True) # test decoder

def train():
    loss = -net['elbo'] # negative ELBO

    global_step = tf.train.get_or_create_global_step()
    lr = tf.train.piecewise_constant(tf.cast(global_step, tf.int32),
            [n_train_batches*args.n_epochs/2], [1e-3, 1e-4])
    train_op = tf.train.AdamOptimizer(lr).minimize(loss,
            global_step=global_step)

    saver = tf.train.Saver(net['weights'])
    logfile = open(os.path.join(savedir, 'train.log'), 'w', 0)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # to run
    train_logger = Accumulator('elbo')
    train_to_run = [train_op, net['elbo']]

    for i in range(args.n_epochs):
        # shuffle the training data
        idx = np.random.choice(range(1000), size=1000, replace=False)
        xtr_ = xtr[idx]

        # run the epoch
        line = 'Epoch %d start, learning rate %f' % (i+1, sess.run(lr))
        print('\n' + line)
        logfile.write('\n' + line + '\n')
        train_logger.clear()
        start = time.time()
        for j in range(n_train_batches):
            bx = xtr_[j*args.batch_size:(j+1)*args.batch_size,:]
            train_logger.accum(sess.run(train_to_run, {x:bx}))
        train_logger.print_(header='train', epoch=i+1,
                time=time.time()-start, logfile=logfile)

    # save the model
    logfile.close()
    saver.save(sess, os.path.join(savedir, 'model'))

def test():
    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights'])
    saver.restore(sess, os.path.join(savedir, 'model'))

    logger = Accumulator('elbo')
    for j in range(n_test_batches):
        bx = xte[j*args.batch_size:(j+1)*args.batch_size,:]
        logger.accum(sess.run(tnet['elbo'], {x:bx}))
    logger.print_(header='test')

def visualize():
    sess = tf.Session()
    saver = tf.train.Saver(tnet['weights'])
    saver.restore(sess, os.path.join(savedir, 'model'))

    def _merge(images, size):
        h, w = images.shape[1], images.shape[2]
        img = np.zeros((h * size[0], w * size[1]))
        for idx, image in enumerate(images):
            i, j = int(idx % size[1]), int(idx / size[1])
            image_ = imresize(image, size=(w, h), interp='bicubic')
            img[j*h:(j+1)*h, i*w:(i+1)*w] = image_
        return img

    # scater-plot
    if args.scatter:
        for i, (x_, y_) in enumerate([(xtr, ytr), (xte, yte)]):
            model = TSNE(learning_rate=10)
            mu = sess.run(tnet['mu'], {x: x_})
            results = model.fit_transform(mu)

            plt.scatter(results[:,0], results[:,1],
                    c=['C%d' % np.argmax(y_[n,:]) for n in range(y_.shape[0])],
                    alpha=0.5)
            flag = 'train' if i == 0 else 'test'
            plt.savefig(savedir + '/scatter_%s.pdf'%flag, format='pdf')
            plt.close()

    # 2-D manifold
    if args.manifold:
        zlim, size = 2, 20
        z_ = np.mgrid[zlim:-zlim:size*1j, zlim:-zlim:size*1j]
        z_ = np.rollaxis(z_, 0, 3).reshape([-1, 2])

        xmap = sess.run(tdenet, {z: z_})
        xmap = xmap.reshape(size*size, 28, 28)
        imsave(savedir + '/manifold.png', _merge(xmap, [size, size]))

if __name__=='__main__':
    if args.mode == 'train':
        train()
    elif args.mode == 'test':
        test()
    elif args.mode == 'visualize':
        visualize()
    else:
        raise ValueError('Invalid mode %s' % args.mode)

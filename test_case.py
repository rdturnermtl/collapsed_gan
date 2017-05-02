import numpy as np
import tensorflow as tf
import pandas as pd

import div_tools as dt

import matplotlib.pyplot as plt
from scipy.special import logit
import cPickle
from scipy.misc import imresize
from scipy.spatial.distance import pdist, squareform


def zeros(shape):
    return tf.Variable(tf.zeros(shape))


def normal(shape, std_dev):
    return tf.Variable(tf.random_normal(shape, stddev=std_dev))


def logdet_tf(S):
    # Is this really not built in to TF??
    ld = 2.0 * tf.reduce_sum(tf.log(tf.diag_part(tf.cholesky(S))))
    return ld


def logdet_lower(L):
    ld = tf.reduce_sum(tf.log(tf.abs(tf.diag_part(L))))
    return ld


def lower_tf(X):
    # Better way to do this??
    return tf.matrix_band_part(X, -1, 0) - tf.matrix_band_part(X, 0, 0)


def learn_gauss_test(train_x, valid_x, batch_size=20):
    sigma_list_obs = np.median(squareform(squareform(pdist(train_x)))) ** 2
    sigma_list_obs = tf.Variable((sigma_list_obs,),
                                 trainable=False, dtype="float")
    sigma_list_latent = tf.Variable((10.0,), trainable=False, dtype="float")

    num_examples, D = train_x.shape
    assert(valid_x.shape == train_x.shape)  # Assume same for now

    train_x_tf = tf.Variable(train_x, trainable=False, dtype="float")
    valid_x_tf = tf.Variable(valid_x, trainable=False, dtype="float")

    # Better to initialize so too small??
    W_dummy = normal((D, D), 0.5 / D)
    # W_tf = lower_tf(W_dummy) + tf.eye(D)
    W_tf = tf.matrix_band_part(W_dummy, -1, 0)
    b_tf = zeros((D,))
    x = tf.placeholder(dtype="float", shape=[batch_size, D])

    samples = tf.matmul(x, W_tf) + b_tf

    # cost of the network, and optimizer for the cost
    cost = tf.reduce_mean(dt.mmd_marg(samples, sigma_list_latent, unbiased=True))
    # Warning: we must change this is W is no longer lower!
    ldw = logdet_lower(W_tf)
    # cost = tf.reduce_mean(dt.nll(samples, ldw))

    optimizer = tf.train.AdamOptimizer().minimize(cost)

    samples_full = tf.matmul(train_x_tf, W_tf) + b_tf

    gen_latent = normal((D, D), 1.0)
    gen_obs = tf.matmul(gen_latent - b_tf, tf.matrix_inverse(W_tf))
    #gen_chk = tf.matmul(gen_obs, W_tf) + b_tf
    #gen_err = tf.reduce_max(tf.abs(gen_latent - gen_chk))
    metric_train = dt.run_all_metrics(train_x_tf, samples_full, ldw,
                                      gen_obs, gen_latent,
                                      sigma_list_obs, sigma_list_latent)
    samples_valid = tf.matmul(valid_x_tf, W_tf) + b_tf
    metric_valid = dt.run_all_metrics(valid_x_tf, samples_valid, ldw,
                                      gen_obs, gen_latent,
                                      sigma_list_obs, sigma_list_latent)

    # initalize all the variables in the model
    init = tf.initialize_all_variables()
    sess = tf.Session()
    sess.run(init)

    W0 = W_tf.eval(session=sess)
    b0 = b_tf.eval(session=sess)
    X0 = np.dot(train_x, W0) + b0[None, :]

    num_iterations = 5000
    iteration_break = 100
    train_hist = []
    valid_hist = []
    train_hist.append({k: np.mean(v.eval(session=sess))
                      for k, v in metric_train.iteritems()})
    valid_hist.append({k: np.mean(v.eval(session=sess))
                      for k, v in metric_valid.iteritems()})
    for i in xrange(num_iterations):
        batch_indices = np.random.choice(num_examples, size=batch_size,
                                         replace=False)
        batch_x = train_x[batch_indices]

        # print out the cost after every 'iteration_break' iterations
        if i % iteration_break == 0:
            curr_cost = sess.run(cost, feed_dict={x: batch_x})
            print 'Cost at iteration ' + str(i+1) + ': ' + str(curr_cost)

            # Re-calculate with np since TF sometimese has trouble here
            logdet_W = np.linalg.slogdet(W_tf.eval(session=sess))[1]
            train_hist.append({k: np.mean(v.eval(session=sess))
                               for k, v in metric_train.iteritems()})
            train_hist[-1]['nll_cmp'] = -logdet_W
            valid_hist.append({k: np.mean(v.eval(session=sess))
                               for k, v in metric_valid.iteritems()})
            valid_hist[-1]['nll_cmp'] = -logdet_W
    
        # optimize the network
        sess.run(optimizer, feed_dict={x: batch_x})
    W_opt = W_tf.eval(session=sess)
    b_opt = b_tf.eval(session=sess)
    X_opt = np.dot(train_x, W_opt) + b_opt[None, :]

    print W_opt
    print b_opt

    return X0, X_opt, train_hist, valid_hist


def load_data(source_file, digit=None, reshuffle=False):
    f = open(source_file, 'rb')
    data = cPickle.load(f)
    f.close()

    if digit is None or digit == 'all':
        x_train = data[0][0]
        x_valid = data[1][0]
        x_test = data[2][0]
    else:
        x_train = data[0][0][data[0][1] == digit, :]
        x_valid = data[1][0][data[1][1] == digit, :]
        x_test = data[2][0][data[2][1] == digit, :]

    if reshuffle:  # To guarantee iid
        n_train, n_valid = x_train.shape[0], x_valid.shape[0]
        x_all = np.concatenate((x_train, x_valid, x_test), axis=0)
        # Multi-dimensional arrays are only shuffled along the first axis
        np.random.shuffle(x_all)
        x_train = x_all[:n_train, :]
        x_valid = x_all[n_train:n_train + n_valid, :]
        x_test = x_all[n_train + n_valid:, :]
    return x_train, x_valid, x_test


def down_sample(X, old_shape, new_shape=(10, 10), jitter=False, warp=False):
    assert(X.shape[1] == np.prod(old_shape))
    assert(np.all(0.0 <= X) and np.all(X < 1.0))
    epsilon = 1.0
    v_range = 256.0

    Y = np.zeros((X.shape[0], np.prod(new_shape)))
    for ii in xrange(X.shape[0]):
        X_sq = np.reshape(X[ii, :], old_shape)
        Y[ii, :] = imresize(X_sq, size=new_shape).ravel()
    assert(np.all(0 <= Y) and np.all(Y <= 255))
    assert(epsilon <= np.min(np.diff(np.unique(Y.ravel()))))

    if jitter:
        Y = Y + epsilon * np.random.rand(Y.shape[0], Y.shape[1])
    Y = Y / v_range  # at least normalize
    assert(np.all(0.0 <= Y) and np.all(Y < 1.0))

    if warp:
        assert(jitter)
        Y = logit(Y)
        assert(np.all(np.isfinite(Y)))
    return Y

if __name__ == '__main__':
    np.random.seed(57421100)

    down_sample_size = 10
    digit = None
    # TODO take as argument
    source_file = '../data/mnist.pkl'

    x_train, x_valid, x_test = load_data(source_file, digit)

    if down_sample_size is not None:
        curr_shape = (28, 28)
        new_shape = (down_sample_size, down_sample_size)
        x_train = down_sample(x_train, curr_shape, new_shape, jitter=True)
        #x_valid = down_sample(x_valid, curr_shape, new_shape, jitter=True)

    R = learn_gauss_test(x_train[:10000, :], x_train[10000:20000, :], batch_size=20)
    X0, X_opt, train_hist, valid_hist = R

    df_valid = pd.DataFrame(valid_hist)
    df_train = pd.DataFrame(train_hist)

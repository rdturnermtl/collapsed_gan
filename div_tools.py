# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np
import tensorflow as tf

# For test only
import div_tools_np as np_test

# TODO also test that jac can change per data point


# Neg. Log Likelihood (KL)


def nll_const(X):
    N, D = X.get_shape().as_list()

    Z_np = 0.5 * D * np.log(2.0 * np.pi)  # Should be known apriori
    C = Z_np * tf.ones((N,))
    assert(C.get_shape().as_list() == [N])
    return C


def nll_fit(X):
    N, _ = X.get_shape().as_list()

    # Is X*X or X**2 faster in tf??
    fit = 0.5 * tf.reduce_sum(X * X, axis=1)
    assert(fit.get_shape().as_list() == [N])
    return fit


def nll_complexity(X, log_det_jac):
    # TODO also implement version that computes this from jac in tf
    N, _ = X.get_shape().as_list()

    penalty = -log_det_jac * tf.ones((N,))
    assert(penalty.get_shape().as_list() == [N])
    return penalty


def nll(X, log_det_jac):
    neg_log_lik = nll_const(X) + nll_fit(X) + nll_complexity(X, log_det_jac)
    return neg_log_lik

# MMD Marginalized (MMD-M)


def mmd_marg_const(X, sigma_list):
    N, D = X.get_shape().as_list()

    C = tf.reduce_sum((sigma_list / (sigma_list + 2.0)) ** (D / 2.0))
    C = C * tf.ones((N,))
    assert(C.get_shape().as_list() == [N])
    return C


def mmd_marg_fit(X, sigma_list):
    N, D = X.get_shape().as_list()
    n_sigma, = sigma_list.get_shape().as_list()

    # dot product of rows with themselves
    neg_g_fit_term = -0.5 * tf.reduce_sum(X * X, axis=1)

    fit = 0.0
    for ii in range(n_sigma):
        Z = -2.0 * ((sigma_list[ii] / (sigma_list[ii] + 1.0)) ** (D / 2.0))
        kernel_val = tf.exp((1.0 / (sigma_list[ii] + 1.0)) * neg_g_fit_term)
        fit += Z * kernel_val
    assert(fit.get_shape().as_list() == [N])
    return fit


def mmd_marg_complexity(X, sigma_list, unbiased=True):
    N, D = X.get_shape().as_list()
    n_sigma, = sigma_list.get_shape().as_list()

    # dot product between all combinations of rows in 'X'
    XX = tf.matmul(X, tf.transpose(X))

    # dot product of rows with themselves
    X2 = tf.reduce_sum(X * X, 1, keep_dims=True)

    # exponent entries of the RBF kernel (without the sigma) for each
    # combination of the rows in 'X'
    # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
    exponent = XX - 0.5 * X2 - 0.5 * tf.transpose(X2)

    penalty = 0.0
    for ii in range(n_sigma):
        if unbiased:
            Z_yy = (1.0 / (N - 1))
            # TIP this could be more efficient by subtracting off const at end
            kernel_val = tf.exp((1.0 / sigma_list[ii]) * exponent) - tf.eye(N)
        else:
            Z_yy = (1.0 / N)
            kernel_val = tf.exp((1.0 / sigma_list[ii]) * exponent)
        penalty += Z_yy * tf.reduce_sum(kernel_val, axis=1)
    assert(penalty.get_shape().as_list() == [N])
    return penalty


def mmd_marg(X, sigma_list, unbiased=True):
    mmd = mmd_marg_const(X, sigma_list) + mmd_marg_fit(X, sigma_list) + \
        mmd_marg_complexity(X, sigma_list, unbiased=unbiased)
    return mmd

# MMD 2-Sample (MMD-2)


def mmd2_const(X, sigma_list, unbiased=True):
    C = tf.reduce_mean(mmd_marg_complexity(X, sigma_list, unbiased=unbiased))
    # Note: could also multiply by ones of size in Y for consistency
    assert(C.get_shape().as_list() == [])
    return C


def mmd2_fit(X, Y, sigma_list):
    M, D = X.get_shape().as_list()
    N, _ = Y.get_shape().as_list()
    assert(Y.get_shape().as_list()[1] == D)
    n_sigma, = sigma_list.get_shape().as_list()

    YX = tf.matmul(Y, tf.transpose(X))
    X2 = tf.reduce_sum(X * X, 1, keep_dims=True)
    Y2 = tf.reduce_sum(Y * Y, 1, keep_dims=True)
    exponent = YX - 0.5 * Y2 - 0.5 * tf.transpose(X2)

    fit = 0.0
    for ii in range(n_sigma):
        Z = -2.0 / M
        kernel_val = tf.exp((1.0 / sigma_list[ii]) * exponent)
        fit += Z * tf.reduce_sum(kernel_val, axis=1)
    assert(fit.get_shape().as_list() == [N])
    return fit

mmd2_complexity = mmd_marg_complexity


def mmd2(X, Y, sigma_list, unbiased=True):
    mmd2_ = mmd2_const(X, sigma_list, unbiased=unbiased) + \
        (mmd2_fit(X, Y, sigma_list) + 
         mmd2_complexity(Y, sigma_list, unbiased=unbiased))
    return mmd2_


def run_all_metrics(data_obs, data_latent, data_log_det_jac,
                    gen_obs, gen_latent,
                    sigma_list_obs, sigma_list_latent):
    metrics = {}

    metrics['nll_fit'] = nll_fit(data_latent)
    metrics['nll_cmp'] = nll_complexity(data_latent, data_log_det_jac)

    metrics['mmdm_fit'] = mmd_marg_fit(data_latent, sigma_list_latent)
    metrics['mmdm_cmp'] = mmd_marg_complexity(data_latent, sigma_list_latent)

    # This metric should be almost the same as mmd marg since mmd marg is just
    # this metric after analytically marginalizing out gen_latent.
    metrics['mmd2l_fit'] = mmd2_fit(gen_latent, data_latent, sigma_list_latent)
    metrics['mmd2l_cmp'] = mmd2_complexity(data_latent, sigma_list_latent)

    metrics['mmd2o_fit'] = mmd2_fit(data_obs, gen_obs, sigma_list_obs)
    metrics['mmd2o_cmp'] = mmd2_complexity(gen_obs, sigma_list_obs)
    return metrics

# Testing

def normal(shape, std_dev=1.0):
    return tf.Variable(tf.random_normal(shape, stddev=std_dev))


def max_err(Y, Y2):
    return np.max(np.abs(Y - Y2))


def run_tests(runs=100, n_sample=500):
    err = []
    for rr in xrange(runs):
        err_curr = 0.0
        unbiased = np.random.rand() <= 0.5

        N_X = np.random.randint(2, 5)
        N_Z = np.random.randint(2, 5)
        D = np.random.randint(1, 5)
        n_sigma = np.random.randint(1, 5)

        X = normal((N_X, D))
        Z = normal((N_Z, D))
        log_det_jac = normal(())
        sigma_list = tf.exp(normal((n_sigma,)))

        init = tf.global_variables_initializer()
        sess = tf.Session()
        sess.run(init)

        Y = nll_const(X)
        Y2 = np_test.nll_const(X.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = nll_fit(X)
        Y2 = np_test.nll_fit(X.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = nll_complexity(X, log_det_jac)
        Y2 = np_test.nll_complexity(X.eval(session=sess),
                                    log_det_jac.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = nll(X, log_det_jac)
        Y2 = np_test.nll(X.eval(session=sess), log_det_jac.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))


        Y = mmd_marg_const(X, sigma_list)
        Y2 = np_test.mmd_marg_const(X.eval(session=sess),
                                    sigma_list.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = mmd_marg_fit(X, sigma_list)
        Y2 = np_test.mmd_marg_fit(X.eval(session=sess),
                                  sigma_list.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = mmd_marg_complexity(X, sigma_list, unbiased=unbiased)
        Y2 = np_test.mmd_marg_complexity(X.eval(session=sess),
                                         sigma_list.eval(session=sess),
                                         unbiased=unbiased)
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = mmd_marg(X, sigma_list, unbiased=unbiased)
        Y2 = np_test.mmd_marg(X.eval(session=sess),
                              sigma_list.eval(session=sess),
                              unbiased=unbiased)
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))


        Y = mmd2_const(X, sigma_list, unbiased=unbiased)
        Y2 = np_test.mmd2_const(X.eval(session=sess),
                                sigma_list.eval(session=sess),
                                unbiased=unbiased)
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = mmd2_fit(X, Z, sigma_list)
        Y2 = np_test.mmd2_fit(X.eval(session=sess), Z.eval(session=sess),
                              sigma_list.eval(session=sess))
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        Y = mmd2(X, Z, sigma_list, unbiased=unbiased)
        Y2 = np_test.mmd2(X.eval(session=sess), Z.eval(session=sess),
                         sigma_list.eval(session=sess),
                         unbiased=unbiased)
        err_curr = max(err_curr, max_err(Y.eval(session=sess), Y2))

        print err_curr
        err.append(err_curr)
    print 'max log10 err: %f' % np.log10(max(err))
    print 'tests done'

if __name__ == '__main__':
    np.random.seed(6742345)
    run_tests()

# Ryan Turner (turnerry@iro.umontreal.ca)
import numpy as np

# For test only
import pandas as pd
import scipy.stats as ss
from scipy.spatial.distance import pdist, squareform

# Neg. Log Likelihood (KL)


def nll_const(X):
    N, D = X.shape

    Z_np = 0.5 * D * np.log(2.0 * np.pi)  # Should be known apriori
    C = Z_np * np.ones((N,))
    assert(C.shape == (N,))
    return C


def nll_fit(X):
    N, _ = X.shape

    fit = 0.5 * np.sum(X * X, axis=1)
    assert(fit.shape == (N,))
    return fit


def nll_complexity(X, log_det_jac):
    N, _ = X.shape

    penalty = -log_det_jac * np.ones((N,))
    assert(penalty.shape == (N,))
    return penalty


def nll(X, log_det_jac):
    neg_log_lik = nll_const(X) + nll_fit(X) + nll_complexity(X, log_det_jac)
    return neg_log_lik

# MMD Marginalized (MMD-M)


def mmd_marg_const(X, sigma_list):
    N, D = X.shape
    sigma_list = np.asarray(sigma_list)

    C = np.sum((sigma_list / (sigma_list + 2.0)) ** (D / 2.0))
    C = C * np.ones((N,))
    assert(C.shape == (N,))
    return C


def mmd_marg_fit(X, sigma_list):
    N, D = X.shape

    # dot product of rows with themselves
    neg_g_fit_term = -0.5 * np.sum(X * X, axis=1)

    fit = 0.0
    for ii in xrange(len(sigma_list)):
        Z = -2.0 * ((sigma_list[ii] / (sigma_list[ii] + 1.0)) ** (D / 2.0))
        kernel_val = np.exp((1.0 / (sigma_list[ii] + 1.0)) * neg_g_fit_term)
        fit += Z * kernel_val
    assert(fit.shape == (N,))
    return fit


def mmd_marg_complexity(X, sigma_list, unbiased=True):
    N, D = X.shape
    assert(N >= 2)

    # dot product between all combinations of rows in 'X'
    XX = np.dot(X, X.T)

    # dot product of rows with themselves
    X2 = np.sum(X * X, 1, keepdims=True)

    # exponent entries of the RBF kernel (without the sigma) for each
    # combination of the rows in 'X'
    # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
    exponent = XX - 0.5 * X2 - 0.5 * X2.T

    penalty = 0.0
    for ii in range(len(sigma_list)):
        if unbiased:
            Z_yy = (1.0 / (N - 1))
            # TIP this could more efficient by subtracting off const at end
            kernel_val = np.exp((1.0 / sigma_list[ii]) * exponent) - np.eye(N)
        else:
            Z_yy = (1.0 / N)
            kernel_val = np.exp((1.0 / sigma_list[ii]) * exponent)
        penalty += Z_yy * np.sum(kernel_val, axis=1)
    assert(penalty.shape == (N,))
    return penalty


def mmd_marg(X, sigma_list, unbiased=True):
    mmd = mmd_marg_const(X, sigma_list) + mmd_marg_fit(X, sigma_list) + \
        mmd_marg_complexity(X, sigma_list, unbiased=unbiased)
    # TIP might want to replace with a clip check operation
    assert(unbiased or np.mean(mmd) >= 0.0)
    return mmd

# MMD 2-Sample (MMD-2)


def mmd2_const(X, sigma_list, unbiased=True):
    C = np.mean(mmd_marg_complexity(X, sigma_list, unbiased=unbiased))
    # Note: could also multiply by ones of size in Y for consistency
    assert(C.shape == ())
    return C


def mmd2_fit(X, Y, sigma_list):
    M, D = X.shape
    N, _ = Y.shape
    assert(Y.shape[1] == D)
    assert(N >= 1 and M >= 1)

    YX = np.dot(Y, X.T)
    X2 = np.sum(X * X, 1, keepdims=True)
    Y2 = np.sum(Y * Y, 1, keepdims=True)
    exponent = YX - 0.5 * Y2 - 0.5 * X2.T

    fit = 0.0
    for ii in range(len(sigma_list)):
        Z = -2.0 / M
        kernel_val = np.exp((1.0 / sigma_list[ii]) * exponent)
        fit += Z * np.sum(kernel_val, axis=1)
    assert(fit.shape == (N,))
    return fit

# These happen to be the same, so no sense in re-writing it
mmd2_complexity = mmd_marg_complexity


def mmd2(X, Y, sigma_list, unbiased=True):
    mmd2_ = mmd2_const(X, sigma_list, unbiased=unbiased) + \
        (mmd2_fit(X, Y, sigma_list) + 
         mmd2_complexity(Y, sigma_list, unbiased=unbiased))
    assert(unbiased or np.mean(mmd2_) >= 0.0)
    return mmd2_

# Statistical testing based on MMD Marg statistic


def mmd_marg_ref_table(N, D, mc_samples, sigma_list, make_order_stats=False):
    ref_tbl = np.zeros((len(sigma_list), mc_samples))
    for ii in xrange(mc_samples):
        X = np.random.randn(N, D)
        for ss_idx, sigma in enumerate(sigma_list):
            # This could be made more efficient by re-using distance matrix
            ref_tbl[ss_idx, ii] = np.mean(mmd_marg(X, (sigma,)))
    if make_order_stats:
        ref_tbl.sort(axis=1)  # Now sort the rows
    return ref_tbl


def mmd_marg_pvalue(X, ref_tbl, sigma_list):
    '''Computes the p-value for each element in sigma list. Can be used for
    parameter selection.'''
    mmd_stat = np.zeros((len(sigma_list), 1))
    for ss_idx, sigma in enumerate(sigma_list):
        # This could be made more efficient by re-using distance matrix
        mmd_stat[ss_idx, 0] = np.mean(mmd_marg(X, (sigma,)))
    pval = np.mean(mmd_stat <= ref_tbl, axis=1)
    return pval


def mmd_marg_pvalue_batches(X, N, n_batches, ref_tbl, sigma_list):
    pval = np.zeros((len(sigma_list), n_batches))
    for rr in xrange(n_batches):
        X_curr = X[np.random.choice(X.shape[0], N, replace=False), :]
        pval[:, rr] = mmd_marg_pvalue(X_curr, ref_tbl, sigma_list)
    return pval


def mmd_marg_mix_pvalue_batches(X, N, n_batches, ref_vec, sigma_list):
    assert(ref_vec.ndim == 1)
    pval = np.zeros((n_batches,))
    for rr in xrange(n_batches):
        X_curr = X[np.random.choice(X.shape[0], N, replace=False), :]
        mmd_stat = np.mean(mmd_marg(X_curr, sigma_list))
        pval[rr] = np.mean(mmd_stat <= ref_vec)
    return pval

# Testing


def mmd_marg_old(X, sigma, normalized=False, unbiased=False):
    M, D = X.shape

    loss = 0.0
    # for each bandwidth parameter, compute the MMD value and add them all
    for i in range(len(sigma)):
        rescale = 1.0 if normalized else (2.0 * np.pi * sigma[i]) ** (D / 2.0)

        # kernel values for each combination of the rows in 'X' 
        Z_xx = (2.0 * np.pi * (sigma[i] + 2.0)) ** (-D / 2.0)
        loss += rescale * Z_xx

        mean_pdf = np.mean(np.prod(ss.norm.pdf(X, loc=0.0, scale=np.sqrt(sigma[i] + 1.0)), axis=1))
        loss += rescale * -2.0 * mean_pdf

        exponent = -0.5 * (squareform(pdist(X)) ** 2)
        if unbiased:
            Z_yy = (1.0 / (M ** 2 - M)) * ((2.0 * np.pi * sigma[i]) ** (-D / 2.0))
            # kernel values for each combination of the rows in 'X' 
            kernel_val = np.exp(1.0 / sigma[i] * exponent) - np.eye(M)
        else:
            Z_yy = (1.0 / M ** 2) * ((2.0 * np.pi * sigma[i]) ** (-D / 2.0))
            kernel_val = np.exp((1.0 / sigma[i]) * exponent)
        loss += rescale * Z_yy * np.sum(kernel_val)
    return loss


def mmd2_old(x, gen_x, sigma):
    batch_size = x.shape[0]
    num_gen = gen_x.shape[0]

    # concatenation of the generated images and images from the dataset
    # first 'N' rows are the generated ones, next 'M' are from the data
    X = np.concatenate([gen_x, x], axis=0)

    # dot product between all combinations of rows in 'X'
    XX = np.dot(X, X.T)

    # dot product of rows with themselves
    X2 = np.sum(X * X, axis=1, keepdims=True)

    # exponent entries of the RBF kernel (without the sigma) for each
    # combination of the rows in 'X'
    # -0.5 * (x^Tx - 2*x^Ty + y^Ty)
    exponent = XX - 0.5 * X2 - 0.5 * X2.T

    # scaling constants for each of the rows in 'X'
    s1 =  (1.0 / num_gen) * np.ones((num_gen, 1))
    s2 = (-1.0 / batch_size) * np.ones((batch_size, 1))
    s =  np.concatenate([s1, s2], axis=0)

    # scaling factors of each of the kernel values, corresponding to the
    # exponent values
    S = np.dot(s, s.T)

    loss = 0.0
    # for each bandwidth parameter, compute the MMD value and add them all
    for i in range(len(sigma)):
        # kernel values for each combination of the rows in 'X' 
        kernel_val = np.exp(1.0 / sigma[i] * exponent)
        loss += np.sum(S * kernel_val)
    return loss


def test_H0(runs=100, n_sample=500):
    pval_marg = np.zeros((runs,))
    pval2 = np.zeros((runs,))
    for rr in xrange(runs):
        N = np.random.randint(2, 10)
        N_X = np.random.randint(2, 10)
        D = np.random.randint(2, 10)

        n_sigma = np.random.randint(1, 5)
        sigma_list = np.exp(np.random.randn(n_sigma))
        sigma_list = np.maximum(0.1, np.minimum(10.0, sigma_list))

        mmdm_u = np.zeros((n_sample,))
        mmd2_u = np.zeros((n_sample,))
        for ii in xrange(n_sample):
            Z = np.random.randn(N, D)
            X = np.random.randn(N_X, D)
            mmdm_u[ii] = np.mean(mmd_marg(Z, sigma_list, unbiased=True))
            # TIP could also warp for arb. distn. in 2-sample case.
            mmd2_u[ii] = np.mean(mmd2(X, Z, sigma_list, unbiased=True))
        _, pval_marg[rr] = ss.ttest_1samp(mmdm_u, 0.0)
        _, pval2[rr] = ss.ttest_1samp(mmd2_u, 0.0)
    return pval_marg, pval2


def run_tests(runs=100, n_sample=500):
    err = []
    pval = np.nan + np.zeros((runs,))
    for rr in xrange(runs):
        err_curr = {}

        # NLL
        N = np.random.randint(1, 5)
        D = np.random.randint(1, 5)
        W = np.random.randn(D, D)

        iW = np.linalg.inv(W)
        S = np.dot(iW.T, iW)
        _, log_det_jac = np.linalg.slogdet(W)

        # Observed
        if np.random.rand() <= 0.5:
            Y = np.random.multivariate_normal(np.zeros(D), S, size=N)
        else:
            Y = np.random.randn(N, D)
        Z = np.dot(Y, W)  # Latent

        nll0 = -ss.multivariate_normal.logpdf(Y, np.zeros(D), S)
        nll1 = nll(Z, log_det_jac)
        err_curr['NLL'] = np.max(np.abs(nll0 - nll1))

        # Now test MMD stuff
        if N == 1:  # MMD doesn't handle N=1
            continue

        n_sigma = np.random.randint(1, 5)
        sigma_list = np.exp(np.random.randn(n_sigma))
        N_X = np.random.randint(2, 5)

        mmdm_u = np.mean(mmd_marg(Z, sigma_list, unbiased=True))
        mmdm_b = np.mean(mmd_marg(Z, sigma_list, unbiased=False))

        # Test new and old marg unbiased
        mmdm_old = mmd_marg_old(Z, sigma_list, normalized=False, unbiased=True)
        err_curr['MMDMu old'] = np.max(np.abs(mmdm_u - mmdm_old))

        # Test new and old marg biased
        mmdm_old = mmd_marg_old(Z, sigma_list,
                                normalized=False, unbiased=False)
        err_curr['MMDMb old'] = np.max(np.abs(mmdm_b - mmdm_old))

        # Check sum under sigma
        mmdm_u2 = np.mean(sum(mmd_marg(Z, (ss,), unbiased=True)
                              for ss in sigma_list))
        err_curr['MMDMu sum'] = np.max(np.abs(mmdm_u2 - mmdm_u))
        mmdm_b2 = np.mean(sum(mmd_marg(Z, (ss,), unbiased=False)
                              for ss in sigma_list))
        err_curr['MMDMb old'] = np.max(np.abs(mmdm_b2 - mmdm_b))

        mmd2_u = np.zeros((n_sample,))
        err_curr['MMD2 sym'] = 0.0
        err_curr['MMD2 old'] = 0.0
        err_curr['MMD2u sum'] = 0.0
        err_curr['MMD2b sum'] = 0.0
        for ii in xrange(n_sample):
            X = np.random.randn(N_X, D)
            mmd2_u[ii] = np.mean(mmd2(X, Z, sigma_list, unbiased=True))

            # Test old vs new 2 sample in biased
            mmd2_biased = np.mean(mmd2(X, Z, sigma_list, unbiased=False))
            mmd2_old_ = mmd2_old(X, Z, sigma_list)
            err_curr['MMD2 old'] = np.maximum(err_curr['MMD2 old'],
                                              np.abs(mmd2_biased - mmd2_old_))

            # Check symmetric
            sym_test = np.mean(mmd2(Z, X, sigma_list, unbiased=True))
            err_curr['MMD2 sym'] = np.maximum(err_curr['MMD2 sym'],
                                              np.abs(mmd2_u[ii] - sym_test))
            sym_test = np.mean(mmd2(Z, X, sigma_list, unbiased=False))
            err_curr['MMD2 sym'] = np.maximum(err_curr['MMD2 sym'],
                                              np.abs(mmd2_biased - sym_test))

            # Check sum under sigma
            mmd2_u2 = np.mean(sum(mmd2(X, Z, (ss,), unbiased=True)
                                  for ss in sigma_list))
            err_curr['MMD2u sum'] = np.max(np.abs(mmd2_u[ii] - mmd2_u2))
            mmd2_b2 = np.mean(sum(mmd2(X, Z, (ss,), unbiased=False)
                                  for ss in sigma_list))
            err_curr['MMD2b sum'] = np.max(np.abs(mmd2_b2 - mmd2_biased))
        # Check marginalization
        _, pval[rr] = ss.ttest_1samp(mmd2_u, mmdm_u)
        err.append(err_curr)
    err = pd.DataFrame(err)
    print err.max(axis=0).to_string()
    print ss.describe(pval, nan_policy='omit')

    pval_marg, pval2 = test_H0(runs=runs, n_sample=n_sample)
    print ss.describe(pval_marg)
    print ss.describe(pval2)
    print 'tests done'

if __name__ == '__main__':
    np.random.seed(823521)
    run_tests()

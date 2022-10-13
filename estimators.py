import numpy as np
import logging
import logging.config
from diffprivlib import tools as dp_tools


def non_private_sparse_mean(X, k, *args, **kwargs):
    d = X.shape[1]
    mean = X.mean(axis=0)
    topk_inds = np.argpartition(X, -k)[-k:]
    estimate = np.zeros(d)
    estimate[topk_inds] = mean[topk_inds]
    return estimate


def clip_coordinates(X, R):
    return np.clip(X, a_min=-R, a_max=R)


def cost_of_privacy_algorithm(X, k, R, epsilon, sigma, *args, **kwargs):
    """
    Implements Alogrithm 3.3 in https://arxiv.org/pdf/1902.04495.pdf.

    R - hyperparameter controlling coordinate-wise clipping;
        assumed to be K * sigma * sqrt(log n)
    """
    def peel_coord(means, scale, selected_coords):
        w = np.random.laplace(scale=scale, size=means.shape)
        noisy_abs = np.abs(means) + w
        # mask already selected coordinates
        noisy_abs[selected_coords] = -np.inf
        return np.argmax(noisy_abs)

    n = X.shape[0]
    lmbda = 2 * (R + sigma * np.sqrt(np.log(n))) / n
    scale = lmbda * k / epsilon  # for basic composition
    # scale = lmbda * np.sqrt(k) / epsilon  # for advanced composition

    clipped_mean = clip_coordinates(X, R + sigma * np.sqrt(np.log(n))).mean(axis=0)

    selected_coords = []
    estimate = np.zeros_like(clipped_mean)
    for i in range(k):
        non_sparse_coord = peel_coord(clipped_mean, scale,
                                      selected_coords)
        estimate[non_sparse_coord] = clipped_mean[non_sparse_coord]
        selected_coords.append(non_sparse_coord)


    w = np.random.laplace(scale=scale, size=clipped_mean.shape)
    
    # add fresh noise to selected coords only
    mask_array = np.zeros_like(w)
    mask_array[selected_coords] = 1.
    mask_array = np.array(mask_array, dtype=bool)
    w[~mask_array] = 0.
    # print(f'w is {w[w != 0]}')
    return estimate + w



def thresholded_peeling(X, k, R, c, epsilon, alpha, sigma, num_bins_uni_est=None):
    """
    Proposed algorithm.
    """
    def private_select_max(means, scale, T, epsilon, selected_coords, d):
        counts = np.sum(np.abs(means) > T, axis=0)
        w = np.random.laplace(scale=scale, size=counts.shape)
        noisy_counts = counts + w
        # mask already selected coordinates
        noisy_counts[selected_coords] = -np.inf
        
        selected_coord = np.argmax(noisy_counts)
        return selected_coord

    n = X.shape[0]
    d = X.shape[1]
    N = n // 2  # we will use half the samples, X_s, to select a coordinate,
    # and the other half, X_m, to estimate the mean of the selected non-sparse coordinates
    X_s = X[0:N, ]
    X_m = X[N: , ]

    scale = k / epsilon  # noise scale for basic composition

    bucket_size = int(epsilon * N / (c * np.log(d) * k))
    T = 2. * sigma / np.sqrt(bucket_size)

    # selecting k coordinates
    selected_coords = []
    assert bucket_size > 0, 'cannot have bucket size be 0'
    num_buckets = N // bucket_size
    samples_in_buckets = np.array_split(X_s, num_buckets)
    bucketed_means = np.array([np.mean(bucket, axis=0) for bucket in samples_in_buckets])

    for round in range(k):
        selected_coord = private_select_max(means=bucketed_means,
                                            scale=scale, T=T,
                                            epsilon=epsilon,
                                            selected_coords=selected_coords,
                                            d=d)
        selected_coords.append(selected_coord)

    sorted_select_coords = sorted(selected_coords)
    X_m_dense = X_m[:, sorted_select_coords]

    dense_estimate = np.zeros(k)
    for i, X_uni_coord in enumerate(X_m_dense.T):
        dense_estimate[i] = univariate_mean_est(X_uni_coord, R,
                                                epsilon, k,
                                                sigma,
                                                num_bins_uni_est)

    estimate = np.zeros(d)
    estimate[sorted_select_coords] = dense_estimate
    return estimate


def univariate_mean_est(X, R, epsilon, k, sigma, num_bins=None):
    n = len(X)
    bound = R + sigma * np.sqrt(np.log(n))
    if num_bins is None:
        num_bins = int(2 * R / (4 * sigma * np.sqrt(np.log(n))))
    if num_bins > 1:
        counts, bin_edges = dp_tools.histogram(X, epsilon/k, bins=num_bins, 
                                               range=(-bound, bound))
        lower_bound, upper_bound = bin_edges[np.argmax(counts) : np.argmax(counts) + 2]
    else:
        lower_bound, upper_bound = -bound, bound
    
    Delta = (upper_bound - lower_bound) / n
    w = np.random.laplace(scale=Delta * k / epsilon)
    return np.clip(X, lower_bound, upper_bound).mean() + w
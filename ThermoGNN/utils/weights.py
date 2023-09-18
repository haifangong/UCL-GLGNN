from collections import Counter

import numpy as np
from scipy.ndimage import convolve1d, gaussian_filter1d
from scipy.signal.windows import triang


def get_lds_kernel_window(kernel, ks, sigma):
    assert kernel in ['gaussian', 'triang', 'laplace']
    half_ks = (ks - 1) // 2
    if kernel == 'gaussian':
        base_kernel = [0.] * half_ks + [1.] + [0.] * half_ks
        kernel_window = gaussian_filter1d(
            base_kernel, sigma=sigma) / max(gaussian_filter1d(base_kernel, sigma=sigma))
    elif kernel == 'triang':
        kernel_window = triang(ks)
    else:
        def laplace(x): return np.exp(-abs(x) / sigma) / (2. * sigma)
        kernel_window = list(map(laplace, np.arange(-half_ks, half_ks + 1))) / max(
            map(laplace, np.arange(-half_ks, half_ks + 1)))

    return kernel_window


def get_bin_idx(x):
    return max(min(int(x * np.float32(1)), 70), -70)


def assign_weights(path):
    
    labels = []
    for line in open(path, 'r'):
        name, _, _, _, value = line.strip().split(' ')
        labels.append(float(value))

    bin_index_per_label = [get_bin_idx(label) for label in labels]
    # calculate empirical (original) label distribution: [Nb,]
    # "Nb" is the number of bins
    Nb = max(bin_index_per_label)

    num_samples_of_bins = dict(Counter(bin_index_per_label))
    emp_label_dist = [num_samples_of_bins.get(i, 0) for i in range(-Nb, Nb)]

    # lds_kernel_window: [ks,], here for example, we use gaussian, ks=5, sigma=2
    lds_kernel_window = get_lds_kernel_window(kernel='gaussian', ks=5, sigma=2)
    # calculate effective label distribution: [Nb,]
    eff_label_dist = convolve1d(np.array(emp_label_dist), weights=lds_kernel_window, mode='constant')

    # Use re-weighting based on effective label distribution, sample-wise weights: [Ns,]
    eff_num_per_label = [eff_label_dist[bin_idx] for bin_idx in bin_index_per_label]
    weights = [np.float32(1 / x) for x in eff_num_per_label]

    return weights

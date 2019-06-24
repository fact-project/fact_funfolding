import numpy as np


def logspace_binning(e_low, e_high, e_ref, n_bins):
    bins = np.logspace(
        np.log10(e_low / e_ref),
        np.log10(e_high / e_ref),
        n_bins + 1,
    )
    return np.append(-np.inf, np.append(bins, np.inf)) * e_ref

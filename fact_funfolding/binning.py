import numpy as np


def logspace_binning(e_low, e_high, e_ref, n_bins):
    return np.logspace(
        np.log10(e_low / e_ref),
        np.log10(e_high / e_ref),
        n_bins
    ) * e_ref

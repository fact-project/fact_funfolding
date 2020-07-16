import numpy as np
import astropy.units as u


def logspace_binning(e_low, e_high, e_ref, n_bins):
    bins = np.logspace(
        np.log10(e_low / e_ref).to(u.dimensionless_unscaled),
        np.log10(e_high / e_ref).to(u.dimensionless_unscaled),
        n_bins + 1,
    )
    return np.append(-np.inf, np.append(bins, np.inf)) * e_ref

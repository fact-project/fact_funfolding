import json
import astropy.units as u
import numpy as np

from fact.analysis.statistics import POINT_SOURCE_FLUX_UNIT


def save_spectrum(outputfile, bins, flux, flux_err, counts, counts_err, g=None, bg=None, add_features=None, **metadata):
    bins = bins.to(u.GeV)
    bin_centers = 0.5 * (bins[:-1] + bins[1:])

    data = {
        'e_low': {
            'value': bins[:-1].value.tolist(),
            'unit': str(bins.unit),
        },
        'e_high': {
            'value': bins[1:].value.tolist(),
            'unit': str(bins.unit),
        },
        'e_center': {
            'value': bin_centers.value.tolist(),
            'unit': str(bins.unit),
        },
        'flux': {
            'value': flux.to(POINT_SOURCE_FLUX_UNIT).value.tolist(),
            'unit': str(POINT_SOURCE_FLUX_UNIT),
        },
        'flux_lower_uncertainty': {
            'value': flux_err[0].to(POINT_SOURCE_FLUX_UNIT).value.tolist(),
            'unit': str(POINT_SOURCE_FLUX_UNIT),
        },
        'flux_upper_uncertainty': {
            'value': flux_err[1].to(POINT_SOURCE_FLUX_UNIT).value.tolist(),
            'unit': str(POINT_SOURCE_FLUX_UNIT),
        },
        'counts': counts.tolist(),
        'counts_lower_uncertainty': counts_err[0].tolist(),
        'counts_upper_uncertainty': counts_err[1].tolist(),
        **metadata
    }

    add_if_not_none(data, g, 'g')
    add_if_not_none(data, bg, 'bg')

    if add_features is not None:
        for key in add_features:
            add_if_not_none(data, add_features[key], key)

    if metadata is not None:
        for key in metadata:
            add_if_not_none(data, metadata[key], key)

    with open(outputfile, 'w') as f:
        json.dump(data, f, indent=4)


def add_if_not_none(result_dict, value, key):
    if value is not None:
        if isinstance(value, u.quantity.Quantity):
            result_dict[key] = {
                'value': value.value if value.isscalar else value.value.tolist(),
                'unit': str(value.unit)
            }
        elif isinstance(value, np.ndarray):
            result_dict[key] = value.tolist() if len(value) > 1 else value
        else:
            result_dict[key] = value


def read_spectrum(inputfile):
    with open(inputfile, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        if isinstance(value, dict) and 'unit' in value.keys():
            data[key] = u.Quantity(value['value'], value['unit'])

    return data

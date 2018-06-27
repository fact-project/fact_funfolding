import json
from fact.analysis.statistics import POINT_SOURCE_FLUX_UNIT
import astropy.units as u


def save_spectrum(outputfile, bins, flux, flux_err, counts, counts_err, **metadata):
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

    with open(outputfile, 'w') as f:
        json.dump(data, f, indent=4)


def read_spectrum(inputfile):
    with open(inputfile, 'r') as f:
        data = json.load(f)

    for key, value in data.items():
        if isinstance(value, dict) and 'unit' in value.keys():
            data[key] = u.Quantity(value['value'], value['unit'])

    return data

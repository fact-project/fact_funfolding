import yaml
import astropy.units as u
from copy import deepcopy


class Config:

    def __init__(self, config_dict):

        self.sample_fraction = config_dict.get('sample_fraction', 1.0)

        self.e_ref = config_dict.get('e_ref', 1 * u.GeV)

        self.threshold = config_dict.get('threshold', 0.85)
        self.theta2_cut = config_dict.get('theta2_cut', 0.025)

        self.e_true_low = config_dict.get('e_true_low', 450 * u.GeV)
        self.e_true_high = config_dict.get('e_true_high', 30 * u.TeV)
        self.n_bins_true = config_dict.get('n_bins_true', 10)

        self.e_est_low = config_dict.get('e_est_low', 400 * u.GeV)
        self.e_est_high = config_dict.get('e_est_high', 30 * u.TeV)
        self.n_bins_est = config_dict.get('n_bins_est', 30)

        self.tau = config_dict.get('tau', None)
        self.background = config_dict.get('background', True)

        self.label = 'FACT Unfolding'

    @staticmethod
    def parse_units(d):
        d = deepcopy(d)
        for k in ('e_ref', 'e_true_low', 'e_true_high', 'e_est_low', 'e_est_high'):
            if k in d:
                d[k] = u.Quantity(d[k]['value'], d[k]['unit'])
        return d

    @classmethod
    def from_yaml(cls, path):
        with open(path, 'r') as f:
            d = yaml.load(f)

        d = cls.parse_units(d)

        return cls(d)

import funfolding as ff
import numpy as np
from fact.io import read_h5py
from irf.collection_area import collection_area
import astropy.units as u
from fact.analysis.statistics import calc_gamma_obstime
from fact.analysis import split_on_off_source_independent
import click

from ..io import save_spectrum
from ..config import Config
from ..binning import logspace_binning

E_PRED = 'gamma_energy_prediction'
E_TRUE = 'corsika_event_header_total_energy'

HEGRA_NORM = 2.79e-7 / (u.m**2 * u.s * u.TeV)


@click.command()
@click.argument('config')
@click.argument('observation_file')
@click.argument('gamma_file')
@click.argument('corsika_file')
@click.argument('output_file')
@click.option('--seed', type=int, default=0)
def main(
    config,
    observation_file,
    gamma_file,
    corsika_file,
    output_file,
    seed,
):
    '''
    unfold fact data
    '''
    random_state = np.random.RandomState(seed)
    np.random.set_state(random_state.get_state())

    config = Config.from_yaml(config)
    e_ref = config.e_ref
    threshold = config.threshold
    theta2_cut = config.theta2_cut

    # define binning in e_est and e_true
    bins_obs = logspace_binning(
        config.e_est_low, config.e_est_high, e_ref, config.n_bins_est
    )
    bins_true = logspace_binning(
        config.e_true_low, config.e_true_high, e_ref, config.n_bins_true
    )

    # read in files
    query = 'gamma_prediction > {} and theta_deg**2 < {}'.format(threshold, theta2_cut)

    gammas = read_h5py(gamma_file, key='events').query(query)

    query = 'gamma_prediction > {}'.format(threshold)
    observations = read_h5py(observation_file, key='events').query(query)

    on, off = split_on_off_source_independent(observations, theta2_cut=theta2_cut)

    observation_runs = read_h5py(observation_file, key='runs')
    obstime = observation_runs.ontime.sum() * u.s

    corsika_events = read_h5py(
        corsika_file,
        key='corsika_events',
        columns=['total_energy'],
    )
    corsika_runs = read_h5py(corsika_file, key='corsika_runs')

    # calculate effective area in given binning
    gamma_obstime = calc_gamma_obstime(
        len(corsika_events) * config.sample_fraction,
        spectral_index=corsika_runs.energy_spectrum_slope.median(),
        max_impact=270 * u.m,
        flux_normalization=HEGRA_NORM,
        e_min=corsika_runs.energy_min.median() * u.GeV,
        e_max=corsika_runs.energy_max.median() * u.GeV,
    )
    a_eff, bin_center, bin_width, a_eff_low, a_eff_high = collection_area(
        corsika_events.total_energy.values,
        gammas[E_TRUE].values,
        impact=270 * u.m,
        bins=bins_true,
        log=False,
        sample_fraction=config.sample_fraction,
    )

    # unfold using funfolding
    X_model = gammas[E_PRED].values
    y_model = gammas[E_TRUE].values

    X_data = on[E_PRED].values

    g_model = np.digitize(X_model, bins_obs.to(u.GeV).value)
    f_model = np.digitize(y_model, bins_true.to(u.GeV).value)

    g_data = np.digitize(X_data, bins_obs.to(u.GeV).value)

    model = ff.model.LinearModel(random_state=random_state)
    model.initialize(digitized_obs=g_model, digitized_truth=f_model)

    vec_g_data, _ = model.generate_vectors(digitized_obs=g_data)
    vec_g_model, vec_f_model = model.generate_vectors(
        digitized_obs=g_model, digitized_truth=f_model
    )

    filled_bins = set(f_model)
    has_underflow = 0 in filled_bins
    has_overflow = len(bins_true) in filled_bins

    if config.background:
        X_bg = off[E_PRED].values
        g_bg = np.digitize(X_bg, bins_obs.to(u.GeV).value)
        vec_g_bg, _ = model.generate_vectors(
            digitized_obs=g_bg,
            obs_weights=np.full(len(g_bg), 0.2)
        )
        model.add_background(vec_g_bg)

    llh = ff.solution.StandardLLH(
        tau=config.tau,
        log_f=True,
        reg_factor_f=1 / a_eff.value,
    )
    llh.initialize(
        vec_g=vec_g_data,
        model=model,
        ignore_n_bins_low=int(has_underflow),
        ignore_n_bins_high=int(has_overflow),
    )

    sol_mcmc = ff.solution.LLHSolutionMCMC(
        n_burn_steps=config.n_burn_steps,
        n_used_steps=config.n_used_steps,
        random_state=random_state,
    )
    sol_mcmc.initialize(llh=llh, model=model)
    sol_mcmc.set_x0_and_bounds(
        x0=np.random.poisson(vec_f_model * obstime / gamma_obstime)
    )

    vec_f_est, sigma_vec_f, sample, probs, autocorr_time = sol_mcmc.fit()

    # throw away under and overflow bin
    if has_overflow:
        vec_f_est = vec_f_est[:-1]
        sigma_vec_f = sigma_vec_f[:, :-1]

    if has_underflow:
        vec_f_est = vec_f_est[1:]
        sigma_vec_f = sigma_vec_f[:, 1:]


    save_spectrum(
        output_file,
        bins_true,
        vec_f_est / a_eff / obstime / bin_width / u.GeV,
        sigma_vec_f / a_eff / obstime / bin_width / u.GeV,
        tau=config.tau,
        label=config.label,

    )


if __name__ == '__main__':
    main()

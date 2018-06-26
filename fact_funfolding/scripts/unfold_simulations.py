import funfolding as ff
import numpy as np
from fact.io import read_h5py
from irf.collection_area import collection_area
import astropy.units as u
import click
from fact.analysis.statistics import calc_gamma_obstime

from ..io import save_spectrum
from ..config import Config
from ..binning import logspace_binning

E_PRED = 'gamma_energy_prediction'
E_TRUE = 'corsika_event_header_total_energy'

HEGRA_NORM = 2.83e-11 / (u.cm**2 * u.s * u.TeV)


@click.command()
@click.argument('config')
@click.argument('gamma_file')
@click.argument('corsika_file')
@click.argument('output_file')
@click.option('--n-test', type=int, default=2500)
@click.option('--seed', type=int, default=0)
def main(
    config,
    gamma_file,
    corsika_file,
    output_file,
    n_test,
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
    corsika_runs = read_h5py(corsika_file, key='corsika_runs')

    query = 'gamma_prediction > {}'.format(threshold)
    corsika_events = read_h5py(
        corsika_file,
        key='corsika_events',
        columns=['total_energy'],
    )

    obstime = calc_gamma_obstime(
        len(corsika_events) * config.sample_fraction,
        spectral_index=corsika_runs.energy_spectrum_slope.median(),
        max_impact=270 * u.m,
        flux_normalization=HEGRA_NORM,
        e_min=corsika_runs.energy_min.median() * u.GeV,
        e_max=corsika_runs.energy_max.median() * u.GeV,
        e_ref=1 * u.TeV,
    ) * (n_test / len(gammas))

    # calculate effective area in given binning
    a_eff, bin_center, bin_width, a_eff_low, a_eff_high = collection_area(
        corsika_events.total_energy.values,
        gammas[E_TRUE].values,
        impact=270 * u.m,
        bins=bins_true.to(u.GeV).value,
        log=False,
        sample_fraction=config.sample_fraction,
    )

    gammas['bin'] = np.digitize(gammas[E_TRUE], bins_true.to(u.GeV).value)
    # split dataframes in train / test set
    gammas['test'] = False
    idx = gammas.sample(n_test, random_state=random_state).index
    gammas.loc[idx, 'test'] = True

    df_test = gammas[gammas.test]
    df_model = gammas[~gammas.test]

    X_model = df_model[E_PRED].values
    y_model = df_model[E_TRUE].values

    X_test = df_test[E_PRED].values
    y_test = df_test[E_TRUE].values

    g_model = np.digitize(X_model, bins_obs.to(u.GeV).value)
    f_model = np.digitize(y_model, bins_true.to(u.GeV).value)

    g_test = np.digitize(X_test, bins_obs.to(u.GeV).value)
    f_test = np.digitize(y_test, bins_true.to(u.GeV).value)

    model = ff.model.LinearModel(random_state=random_state)
    model.initialize(digitized_obs=g_model, digitized_truth=f_model)

    vec_g_test, vec_f_test = model.generate_vectors(
        digitized_obs=g_test, digitized_truth=f_test
    )
    vec_g_model, vec_f_model = model.generate_vectors(
        digitized_obs=g_model, digitized_truth=f_model
    )

    filled_bins = set(f_model)
    has_underflow = 0 in filled_bins
    has_overflow = len(bins_true) in filled_bins

    llh = ff.solution.StandardLLH(
        tau=config.tau,
        log_f=True,
        reg_factor_f=1 / a_eff.value if config.tau else None,
    )
    llh.initialize(
        vec_g=vec_g_test,
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
        x0=np.random.poisson(vec_f_test)
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
        vec_f_est / a_eff / bin_width / u.GeV / obstime,
        sigma_vec_f / a_eff / bin_width / u.GeV / obstime,
        tau=config.tau,
        label=config.label,
    )


if __name__ == '__main__':
    main()

import funfolding as ff
import numpy as np
from fact.io import read_h5py
import matplotlib.pyplot as plt
from irf.collection_area import collection_area
import astropy.units as u
from fact.analysis.statistics import calc_gamma_obstime
from fact.analysis import split_on_off_source_independent
from fact.analysis.statistics import power_law, curved_power_law, POINT_SOURCE_FLUX_UNIT
from spectrum_io import save_spectrum
import click

SAMPLE_FRACTION = 0.75

E_PRED = 'gamma_energy_prediction'
E_TRUE = 'corsika_event_header_total_energy'

HEGRA_NORM = 2.79e-7 / (u.m**2 * u.s * u.TeV)

THRESHOLD = 0.85
THETA2_CUT = 0.025

BINS_OBS = np.logspace(np.log10(400), np.log10(30e3), 31) * u.GeV
BINS_TRUTH = np.logspace(np.log10(450), np.log10(30e3), 11) * u.GeV


@click.command()
@click.argument('data_file')
@click.argument('gamma_file')
@click.argument('corsika_file')
@click.argument('output_file')
@click.option('--tau', type=float, help='Regularization Parameter')
@click.option(
    '--background/--no-background', default=True,
    help='Weither to take background into account'
)
@click.option('--label', help='A label to add to the plot', default='funfolding')
def main(
    data_file,
    gamma_file,
    corsika_file,
    output_file,
    tau,
    background,
    label,
):
    '''
    unfold fact data
    '''
    query = f'gamma_prediction > {THRESHOLD} and theta_deg**2 < {THETA2_CUT}'

    gammas = read_h5py(gamma_file, key='events').query(query)

    query = f'gamma_prediction > {THRESHOLD}'
    crab = read_h5py(data_file, key='events').query(query)

    print(gammas[E_TRUE].describe())
    print(crab[E_PRED].describe())

    on, off = split_on_off_source_independent(crab, theta2_cut=THETA2_CUT)

    crab_runs = read_h5py(data_file, key='runs')
    obstime = crab_runs.ontime.sum() * u.s

    corsika_events = read_h5py(
        corsika_file,
        key='corsika_events',
        columns=['total_energy'],
    )
    corsika_runs = read_h5py(corsika_file, key='corsika_runs')
    gamma_obstime = calc_gamma_obstime(
        len(corsika_events) * SAMPLE_FRACTION,
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
        bins=BINS_TRUTH,
        log=False,
        sample_fraction=SAMPLE_FRACTION,
    )

    plt.errorbar(
        bin_center,
        a_eff.value,
        yerr=(
            a_eff.value - a_eff_low.value,
            a_eff_high.value - a_eff.value,
        ),
        xerr=bin_width / 2,
        ls=''
    )
    plt.yscale('log')
    plt.xscale('log')
    plt.show()

    X_model = gammas[E_PRED].values
    y_model = gammas[E_TRUE].values

    X_data = on[E_PRED].values

    g_model = np.digitize(X_model, BINS_OBS.to(u.GeV).value)
    f_model = np.digitize(y_model, BINS_TRUTH.to(u.GeV).value)

    g_data = np.digitize(X_data, BINS_OBS.to(u.GeV).value)

    model = ff.model.LinearModel()
    model.initialize(digitized_obs=g_model, digitized_truth=f_model)

    vec_g_data, _ = model.generate_vectors(digitized_obs=g_data)
    vec_g_model, vec_f_model = model.generate_vectors(
        digitized_obs=g_model, digitized_truth=f_model
    )

    filled_bins = set(f_model)
    has_underflow = 0 in filled_bins
    has_overflow = len(BINS_TRUTH) in filled_bins

    if background:
        X_bg = off[E_PRED].values
        g_bg = np.digitize(X_bg, BINS_OBS.to(u.GeV).value)
        vec_g_bg, _ = model.generate_vectors(
            digitized_obs=g_bg,
            obs_weights=np.full(len(g_bg), 0.2)
        )
        model.add_background(vec_g_bg)

    llh = ff.solution.StandardLLH(
        tau=tau,
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
        n_burn_steps=10000,
        n_used_steps=1000,
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
        BINS_TRUTH,
        vec_f_est / a_eff / obstime / bin_width / u.GeV,
        sigma_vec_f / a_eff / obstime / bin_width / u.GeV,
        tau=tau,
        label=label,

    )


if __name__ == '__main__':
    main()

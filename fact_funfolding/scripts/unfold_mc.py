import funfolding as ff
import numpy as np
from fact.io import read_h5py
import matplotlib.pyplot as plt
from irf.collection_area import collection_area
import astropy.units as u


test_fraction = 0.05
df = read_h5py('data/gamma_test_dl3.hdf5', key='events')
df = df[(df.gamma_prediction > 0.85) & (df.theta_deg**2 < 0.025)].copy()

corsika_events = read_h5py(
    'data/gamma_gustav_werner_corsika.hdf5',
    key='corsika_events',
    columns=['total_energy'],
)

# split dataframes in train / test set
df['test'] = False
df.loc[df.sample(frac=test_fraction).index, 'test'] = True


df_test = df[df.test]
df_model = df[~df.test]


e_pred_col = 'gamma_energy_prediction'
e_true_col = 'corsika_event_header_total_energy'

X_model = df_model[e_pred_col].values
y_model = df_model[e_true_col].values

X_test = df_test[e_pred_col].values
y_test = df_test[e_true_col].values


bins_obs = np.logspace(np.log10(400), np.log10(30e3), 31)
bins_truth = np.logspace(np.log10(300), np.log10(30e3), 11)


g_model = np.digitize(X_model, bins_obs)
f_model = np.digitize(y_model, bins_truth)

g_test = np.digitize(X_test, bins_obs)
f_test = np.digitize(y_test, bins_truth)

model = ff.model.LinearModel()
model.initialize(digitized_obs=g_model, digitized_truth=f_model)

vec_g_test, vec_f_test = model.generate_vectors(
    digitized_obs=g_test,
    digitized_truth=f_test,
)

vec_g_model, vec_f_model = model.generate_vectors(
    digitized_obs=g_model, digitized_truth=f_model
)

x = 0.5 * (bins_truth[0:-1] + bins_truth[1:])
xerr = (
    x - bins_truth[:-1],
    bins_truth[1:] - x,
)

a_eff, x, dx, a_eff_low, a_eff_high = collection_area(
    corsika_events.total_energy.values,
    df[e_true_col].values,
    impact=270 * u.m,
    bins=bins_truth,
    log=False,
)

plt.figure()
plt.errorbar(x, a_eff.value, xerr=xerr, yerr=((a_eff - a_eff_low).value, (a_eff_high - a_eff).value), ls='')
plt.xscale('log')
plt.yscale('log')
plt.show()

plt.figure()
plt.errorbar(
    x,
    vec_f_test[1:-1] / a_eff.value,
    xerr=xerr,
    label='truth',
    ls=''
)

tau = None

llh = ff.solution.StandardLLH(tau=tau, log_f=True)
llh.initialize(vec_g=vec_g_test, model=model)

sol_mcmc = ff.solution.LLHSolutionMCMC(
    n_burn_steps=5000,
    n_used_steps=5000,
)
sol_mcmc.initialize(llh=llh, model=model)
sol_mcmc.set_x0_and_bounds(x0=np.random.poisson(test_fraction * vec_f_model))

vec_f_est, sigma_vec_f, sample, probs, autocorr_time = sol_mcmc.fit()

y_err_low = vec_f_est - sigma_vec_f[0]
y_err_high = sigma_vec_f[1] - vec_f_est

plt.errorbar(
    x,
    vec_f_est[1:-1] / a_eff.value,
    xerr=xerr,
    yerr=(y_err_low[1:-1] / a_eff.value, y_err_high[1:-1] / a_eff.value),
    label=f'tau = {tau}',
    ls='',
)

plt.legend()
plt.xscale('log')
plt.yscale('log')
plt.xlabel('E / GeV')
plt.ylabel('N')
plt.savefig('funfolding_f.pdf', dpi=300)

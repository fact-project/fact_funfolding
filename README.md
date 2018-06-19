# FACT funfolding

Unfold FACT spectra using https://github.com/tudo_astroparticlephysics/funfolding

## Installation

Install the current master of `emcee`, then install fact_funfolding
```
$ pip install https://github.com/dfm/emcee/archive/master.tar.gz
$ pip install https://github.com/fact-project/fact_funfolding
```


## Unfold a MC test set

```
$ fact_unfold_simulations \
  examples/config.yaml \
  data/gamma_test_dl3.hdf5 \
  data/gamma_gustav_werner_corsika.hdf5 \
  test_mc.yaml \
  --n-test=2200
```

## Unfold observations

```
$ fact_unfold_observations \
  examples/config.yaml \
  data/crab_dl3.hdf5 \
  data/gamma_test_dl3.hdf5 \
  data/gamma_gustav_werner_corsika.hdf5 \
  test_data.yaml
```

## Plot results

Plot the unfolding results

```
fact_plot_spectrum \
  test_data.yaml \
  -p examples/magic_jheap2015.yaml \
  -p examples/hegra.yaml \
  -o spectrum.pdf
```

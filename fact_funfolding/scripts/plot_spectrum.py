from fact.analysis.statistics import power_law, curved_power_law, POINT_SOURCE_FLUX_UNIT
import astropy.units as u
import matplotlib.pyplot as plt
import click
import numpy as np
from spectrum_io import read_spectrum
import yaml
import os


def hegra_crab(E):
    HEGRA_NORM = 2.79e-7 / (u.m**2 * u.s * u.TeV)
    HEGRA_INDEX = -2.59
    return power_law(E, HEGRA_NORM, HEGRA_INDEX)


def magic_crab(E):
    MAGIC_NORM = 3.23e-11 / u.cm**2 / u.s / u.TeV
    MAGIC_A = -2.47
    MAGIC_B = -0.24
    return curved_power_law(E, MAGIC_NORM, MAGIC_A, MAGIC_B, e_ref=1 * u.TeV)


def magic_crab_performance(E):
    MAGIC_NORM = 5.8e-10 / u.cm**2 / u.s / u.TeV
    MAGIC_A = -2.32
    MAGIC_B = -0.13
    return curved_power_law(E, MAGIC_NORM, MAGIC_A, MAGIC_B, e_ref=300 * u.GeV)


def plot_gammapy(e_plot, result):

    norm = result['parameters'][0]
    norm = u.Quantity(norm['value'], norm['unit'])

    a = result['parameters'][2]
    a = -u.Quantity(a['value'], a['unit'])

    b = result['parameters'][3]
    b = -u.Quantity(b['value'], b['unit'])

    e_ref = result['parameters'][1]
    e_ref = u.Quantity(e_ref['value'], e_ref['unit'])

    return curved_power_law(e_plot, norm, a, b, e_ref)




@click.command()
@click.option('--fit-result', multiple=True, type=click.Path(exists=True, dir_okay=False))
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False))
@click.argument('spectra', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(fit_result, outputfile, spectra):
    for zorder, spectrum in enumerate(spectra, start=2):
        data = read_spectrum(spectrum)

        e_min = 0.5 * min(data['e_low'].to(u.GeV).value)
        e_max = 2 * max(data['e_high'].to(u.GeV).value)
        e_plot = np.logspace(np.log10(e_min), np.log10(e_max), 250) * u.GeV

        x = data['e_center'].to(u.GeV).value
        plt.errorbar(
            x,
            x**2 * data['flux'].to(POINT_SOURCE_FLUX_UNIT).value,
            xerr=(
                (data['e_center'] - data['e_low']).value,
                (data['e_high'] - data['e_center']).value,
            ),
            yerr=(
                x**2 * (data['flux'] - data['flux_lower_uncertainty']).to(POINT_SOURCE_FLUX_UNIT).value,
                x**2 * (data['flux_upper_uncertainty'] - data['flux']).to(POINT_SOURCE_FLUX_UNIT).value,
            ),
            label=data['label'],
            ls='',
            capsize=2,
            zorder=zorder,
        )

    for inputfile in fit_result:
        with open(inputfile) as f:
            fit_result = yaml.load(f)

        name, _ = os.path.splitext(inputfile)

        plt.plot(
            e_plot.to(u.GeV).value,
            e_plot**2 * plot_gammapy(e_plot, fit_result).to(POINT_SOURCE_FLUX_UNIT).value,
            label=name.replace('_', ' '),
        )

    # plt.plot(
    #     e_plot.to(u.GeV).value,
    #     hegra_crab(e_plot).to(POINT_SOURCE_FLUX_UNIT).value,
    #     label='HEGRA',
    #     zorder=0,
    #     color='lightgray'
    # )

    plt.plot(
        e_plot.to(u.GeV).value,
        (e_plot**2 * magic_crab(e_plot)).to(u.GeV**2 * POINT_SOURCE_FLUX_UNIT).value,
        label='MAGIC JHEAP 2015 5-6',
        zorder=0,
        color='darkgray'
    )

    plt.plot(
        e_plot.to(u.GeV).value,
        (e_plot**2 * magic_crab_performance(e_plot)).to(u.GeV**2 * POINT_SOURCE_FLUX_UNIT).value,
        label='MAGIC APP 2012 35.7',
        zorder=0,
        color='lightgray'
    )

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E / GeV')
    plt.ylabel(f'$E^2 \cdot Φ \,\,/\,\, {{}}${(u.GeV**2 * POINT_SOURCE_FLUX_UNIT):latex_inline}')
    print('saving plot')
    plt.savefig(outputfile, dpi=300)


if __name__ == '__main__':
    main()

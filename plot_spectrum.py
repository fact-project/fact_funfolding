from fact.analysis.statistics import power_law, curved_power_law, POINT_SOURCE_FLUX_UNIT
import astropy.units as u
import matplotlib.pyplot as plt
import click
import numpy as np
from spectrum_io import read_spectrum


def hegra_crab(E):
    HEGRA_NORM = 2.79e-7 / (u.m**2 * u.s * u.TeV)
    HEGRA_INDEX = -2.59
    return power_law(E, HEGRA_NORM, HEGRA_INDEX)


def magic_crab(E):
    MAGIC_NORM = 3.23e-11 / u.cm**2 / u.s / u.TeV
    MAGIC_A = -2.47
    MAGIC_B = -0.24
    return curved_power_law(E, MAGIC_NORM, MAGIC_A, MAGIC_B, e_ref=1 * u.TeV)


@click.command()
@click.argument('outputfile', type=click.Path(exists=False, dir_okay=False))
@click.argument('spectra', nargs=-1, type=click.Path(exists=True, dir_okay=False))
def main(outputfile, spectra):
    for zorder, spectrum in enumerate(spectra, start=2):
        data = read_spectrum(spectrum)

        e_min = 0.5 * min(data['e_low'].to(u.GeV).value)
        e_max = 2 * max(data['e_high'].to(u.GeV).value)
        e_plot = np.logspace(np.log10(e_min), np.log10(e_max), 250) * u.GeV

        plt.errorbar(
            data['e_center'].to(u.GeV).value,
            data['flux'].to(POINT_SOURCE_FLUX_UNIT).value,
            xerr=(
                (data['e_center'] - data['e_low']).value,
                (data['e_high'] - data['e_center']).value,
            ),
            yerr=(
                (data['flux'] - data['flux_lower_uncertainty']).to(POINT_SOURCE_FLUX_UNIT).value,
                (data['flux_upper_uncertainty'] - data['flux']).to(POINT_SOURCE_FLUX_UNIT).value,
            ),
            label=data['label'],
            ls='',
            capsize=2,
            zorder=zorder,
        )

    plt.plot(
        e_plot.to(u.GeV).value,
        hegra_crab(e_plot).to(POINT_SOURCE_FLUX_UNIT).value,
        label='HEGRA',
        zorder=0,
        color='lightgray'
    )

    plt.plot(
        e_plot.to(u.GeV).value,
        magic_crab(e_plot).to(POINT_SOURCE_FLUX_UNIT).value,
        label='MAGIC',
        zorder=0,
        color='darkgray'
    )

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('E / GeV')
    plt.ylabel(f'$Φ \,\,/\,\, {{}}${POINT_SOURCE_FLUX_UNIT:latex_inline}')
    print('saving plot')
    plt.savefig(outputfile, dpi=300)


if __name__ == '__main__':
    main()

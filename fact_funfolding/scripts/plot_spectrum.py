from fact.analysis.statistics import power_law, curved_power_law, POINT_SOURCE_FLUX_UNIT
import astropy.units as u
import matplotlib.pyplot as plt
import click
import numpy as np
import yaml


from ..io import read_spectrum


def flux_publication_result(e_plot, result):
    func = result['function']
    assert func in ('power_law', 'log_parabola', 'exponential_cutoff'), 'Spectral function not understood'

    norm = u.Quantity(**result['phi_0'])
    e_ref = u.Quantity(**result['e_ref'])

    if func == 'log_parabola':
        a = result['a']
        b = result['b']

        return curved_power_law(
            e_plot, flux_normalization=norm, a=a, b=b, e_ref=e_ref
        )

    index = result['spectral_index']
    power = power_law(e_plot, norm, index, e_ref)

    if func == 'exponential_cutoff':
        e_cutoff = u.Quantity(**result['e_cutoff'])
        power *= np.exp(- e_plot / e_cutoff)

    return power


def flux_gammapy_fit_result(e_plot, result):
    parameters = {p['name']: p for p in result['parameters']}

    norm = parameters['amplitude']
    norm = u.Quantity(norm['value'], norm['unit'])

    a = parameters['alpha']
    a = -u.Quantity(a['value'], a['unit'])

    b = parameters['beta']
    b = -u.Quantity(b['value'], b['unit'])

    e_ref = parameters['reference']
    e_ref = u.Quantity(e_ref['value'], e_ref['unit'])

    return curved_power_law(e_plot, norm, a, b, e_ref)


@click.command()
@click.option(
    '-g', '--gammapy-fit-result',
    multiple=True,
    type=click.Tuple([str, click.Path(exists=True, dir_okay=False)]),
    help='A label and a yaml fit result as produced by gammapy'
)
@click.option(
    '-p', '--publication-result',
    multiple=True,
    type=click.Path(exists=True, dir_okay=False),
    help=('A fit result in yaml format. Must contain the keys '
          '"function" "e_ref", "phi_0" as well as "a" and "b" for function=log_parabola'
          'and "spectral_index" for function=power_law. May also contain label')

)
@click.option('-o', '--outputfile', type=click.Path(exists=False, dir_okay=False))
@click.option('--e2', is_flag=True, help='Scale by E²')
@click.argument(
    'spectra',
    required=True,
    nargs=-1,
    type=click.Path(exists=True, dir_okay=False)
)
def main(gammapy_fit_result, publication_result, spectra, outputfile, e2):

    for zorder, spectrum in enumerate(spectra, start=3):
        data = read_spectrum(spectrum)

        e_min = 0.5 * min(data['e_low'].to(u.GeV).value)
        e_max = 2 * max(data['e_high'].to(u.GeV).value)
        e_plot = np.logspace(np.log10(e_min), np.log10(e_max), 250) * u.GeV

        x = data['e_center'].to(u.GeV).value
        if e2:
            scale = x**2
        else:
            scale = 1

        yerr = u.Quantity([
            (data['flux'] - data['flux_lower_uncertainty']),
            (data['flux_upper_uncertainty'] - data['flux'])
        ]).to(POINT_SOURCE_FLUX_UNIT).value
        plt.errorbar(
            x,
            scale * data['flux'].to(POINT_SOURCE_FLUX_UNIT).value,
            xerr=(
                (data['e_center'] - data['e_low']).value,
                (data['e_high'] - data['e_center']).value,
            ),
            yerr=scale * yerr,
            label=data['label'],
            ls='',
            zorder=zorder,
        )

    if e2:
        scale = e_plot**2
    else:
        scale = 1.0

    for label, inputfile in gammapy_fit_result:
        with open(inputfile) as f:
            fit_result = yaml.load(f)

        y = flux_gammapy_fit_result(e_plot, fit_result).to(POINT_SOURCE_FLUX_UNIT)
        plt.plot(
            e_plot.to(u.GeV).value,
            scale * y.value,
            label=label,
        )

    for inputfile in publication_result:
        with open(inputfile) as f:
            fit_result = yaml.load(f)

        y = flux_publication_result(e_plot, fit_result).to(POINT_SOURCE_FLUX_UNIT)
        plt.plot(
            e_plot.to(u.GeV).value,
            scale * y.value,
            label=fit_result.get('label'),
            color=fit_result.get('color'),
        )

    label = '\Phi \,\,/\,\, {{}}(${:latex_inline}$)$'
    if e2:
        label = '$E^2 \cdot ' + label.format(u.GeV**2 * POINT_SOURCE_FLUX_UNIT)
    else:
        label = '$' + label.format(POINT_SOURCE_FLUX_UNIT)

    plt.legend()
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel(r'$E \,\,/\,\, \mathrm{GeV}$')
    plt.ylabel(label)
    plt.tight_layout(pad=0)
    if outputfile:
        plt.savefig(outputfile, dpi=300)
    else:
        plt.show()


if __name__ == '__main__':
    main()

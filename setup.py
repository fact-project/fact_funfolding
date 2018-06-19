from setuptools import setup, find_packages


setup(
    name='fact_funfolding',
    description='Commandline utility to use funfolding on fact data',
    version='0.1.1',
    author='Maximilian Nöthe',
    author_email='maximilian.noethe@tu-dortmund.de',
    packages=find_packages(),
    install_requires=[
        'funfolding',
        'pyfact',
        'pyyaml',
        'numpy',
        'astropy',
        'irf',
    ],
    entry_points={
        'console_scripts': [
            'fact_unfold_observations = fact_funfolding.scripts.unfold_observations:main',
            'fact_unfold_simulations = fact_funfolding.scripts.unfold_simulations:main',
            'fact_plot_spectrum = fact_funfolding.scripts.plot_spectrum:main',
        ]
    }
)

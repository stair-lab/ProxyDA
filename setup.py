import os
import sys
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(here, "KPLA"))
from version import __version__

setup(
    name="KPLA",
    description="kernel proxy methods for domain adaptation",
    author="Katherine Tsai, Olawale Salaudeen, Nicole Chiou",
    version=__version__,
    packages=find_packages(),
    install_requires=[
        "cosde",
        "cvxpy >= 1.4.1",
        "cvxopt >= 1.3.2",
        "jax >= 0.4.20",
        "jaxlib >= 0.4.25",
        "latent_shift_adaptation >= 0.1.0",
        "pandas >= 2.0.3",
        "matplotlib >= 3.7.2",
        "numpy >= 1.24.3",
        "scikit-image >= 0.22.0",
        "scikit-learn >= 1.3.0",
        "scipy >= 1.11.2",
        "tqdm >= 4.66.1",
        "tensorflow >= 2.11.0",
    ],
    license_files=("LICENSE"),
)

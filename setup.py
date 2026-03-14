from setuptools import setup, find_packages

setup(
    name="thermoelectric_prediction",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.3.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
)

from setuptools import setup, find_packages

setup(
    name="football-season-predictor",
    version="0.1.0",
    description="A package to predict LaLiga and EPL season outcomes using a Random Forest model.",
    author="Aryan Singh",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.24.0",
    ],
)
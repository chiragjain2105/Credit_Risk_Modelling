from setuptools import find_packages, setup
from typing import List

setup(
    name="Credit_Risk_Modelling",
    version="0.0.1",
    author="ChiragJain",
    author_email="jainchiragcj2105@gmail.com",
    install_requires=["scikit-learn","pandas","numpy"],
    packages=find_packages()
)
# !/usr/bin/env python
from importlib import import_module
from setuptools import find_packages, setup

setup(
    name="symmetry_dp",
    version=0.1,
    author="Various Contributors",
    author_email="jesseperla@gmail.com",
    description="Symmetry and Dynamic Programming",
    license="MIT",
    url="",
    packages=find_packages(),
    # long_description=read("README.md"),
    install_requires=[],
    tests_require=[
        "pytest",
        "pytest-cov",
    ],
    platforms="any",
    python_requires=">=3.8",
)

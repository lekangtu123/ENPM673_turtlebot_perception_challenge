import os
from setuptools import find_packages
from setuptools import setup

setup(
    name='enpm673_module',
    version='0.0.0',
    packages=find_packages(
        include=('enpm673_module', 'enpm673_module.*')),
)

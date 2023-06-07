import re
from setuptools import setup, find_packages
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'ngym_priors'))
from version import VERSION


if sys.version_info.major != 3:
    print('This Python is only compatible with Python 3, but you are running '
          'Python {}. The installation will likely fail.'.format(sys.version_info.major))


setup(name='ngym_priors',
      packages=[package for package in find_packages()
                if package.startswith('ngym_priors')],
      install_requires=[
          'numpy',
          'gym',
          'matplotlib',
      ],
      description='NeuroGym-Priors: Gym-style Priors tasks',
      author='Manuel Molano',
      url='https://github.com/gyyang/neurogym',
      version=VERSION)


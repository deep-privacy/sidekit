#!/usr/bin/env python

from distutils.core import setup

setup(name='sidekit',
      version='1.0',
      install_requires=[
          "matplotlib>=3.4.3",
          "SoundFile>=0.10.3.post1",
          "PyYAML>=5.4.1",
          "h5py>=3.2.1",
          "ipython>=7.27.0",
          "scikit-learn>=0.24.2",
          "pandas>=1.3.3",
          "torch>=1.7.0",
        ],
     )

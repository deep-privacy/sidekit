#!/usr/bin/env python3

import os
from distutils.core import setup

_here = os.path.abspath(os.path.dirname(__file__))

with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(name='sidekit',
      version="2.0.0",
  long_description_content_type="text/markdown",
    description=(
        "SIDEKIT is an open source package for Speaker and Language recognition."
    ),
    long_description=long_description,
    author="Anthony Larcher",
    url="https://git-lium.univ-lemans.fr/Larcher/sidekit",
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
    scripts = [
        'sidekit/bin/compute_spk_cosine.py',
        'sidekit/bin/extract_xvectors.py',
        'tools/compute_metrics.py',
    ]
)

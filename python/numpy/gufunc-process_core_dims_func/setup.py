# This file follows example shown at
# https://setuptools.pypa.io/en/latest/userguide/ext_modules.html

import numpy as np
from setuptools import setup, Extension


experiment = Extension(name='experiment',
                       sources=['experiment.c'],
                       include_dirs=[np.get_include()])

setup(ext_modules=[experiment])

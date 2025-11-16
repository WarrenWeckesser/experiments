import numpy as np
from setuptools import setup, Extension

module = Extension('iter_examples',
                   sources=['iter_examples.c'],
                   include_dirs=[np.get_include()])

setup(
    name='iter_examples',
    version='0.1',
    ext_modules=[module]
)

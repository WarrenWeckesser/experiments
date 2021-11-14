# This file follows example shown at
# https://docs.python.org/3/extending/building.html#building-c-and-c-extensions-with-distutils

from distutils.core import setup, Extension
from os.path import join


funccall = Extension('funccall',
                     sources=[join('src', 'funccallmodule.c')])

setup(name='funccall',
      version='0.1',
      ext_modules=[funccall])

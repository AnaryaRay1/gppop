#!/usr/bin/env python
import setuptools  
from numpy.distutils.core import setup, Extension

lib = Extension(name='gppop.lib.toeplitz_cholesky', sources=['src/lib/toeplitz_cholesky.f90'])
if __name__ == "__main__":
    setup(scripts=['bin/run_gppop'],ext_modules = [lib])
from setuptools import Extension
from setuptools import setup
from Cython.Build import cythonize
import numpy as np

extension = Extension(
    '*', ['median_filter.pyx'],
    include_dirs=[np.get_include()],
    define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
)

setup(ext_modules=cythonize([extension]))

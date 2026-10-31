from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

ext = Pybind11Extension('fastticks._core', ['src/fastticks/_core.cpp'], cxx_std=17,
                        extra_compile_args=['-O3'], depends=['src/fastticks/window.hpp'])
setup(name='fastticks', version='0.2.0', packages=['fastticks'], package_dir={'': 'src'},
      ext_modules=[ext], cmdclass={'build_ext': build_ext}, python_requires='>=3.8',
      install_requires=['numpy'], zip_safe=False)

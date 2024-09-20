import os
from setuptools import setup, Extension
from pybind11.setup_helpers import Pybind11Extension, build_ext
import subprocess
import sys
from setuptools.command.build_ext import build_ext

class BuildExt(build_ext):
    def build_extensions(self):
        if '-fopenmp' in self.compiler.compiler_so:
            for ext in self.extensions:
                ext.extra_compile_args = ['-fopenmp']
                ext.extra_link_args = ['-fopenmp']
        super().build_extensions()

def get_python_lib():
    return subprocess.check_output(['python3-config', '--ldflags']).decode('utf-8').strip()

extra_compile_args = []
extra_link_args = []

if sys.platform == 'darwin':  # macOS
    extra_compile_args = ['-Xpreprocessor', '-fopenmp']
    extra_link_args = ['-lomp']
elif sys.platform == 'win32':  # Windows
    extra_compile_args = ['/openmp']
else:  # Linux and others
    extra_compile_args = ['-fopenmp']
    extra_link_args = ['-fopenmp']

ext_modules = [
    Pybind11Extension(
        "analyzer_wrapper",
        ["NeuroSpector-main/src/analyzer_wrapper/analyzer_wrapper.cc"],
        include_dirs=["NeuroSpector-main/src",
                      "NeuroSpector-main/src/analyzer",
                      "NeuroSpector-main/src/common",
                      "NeuroSpector-main/src/utility",
                      "NeuroSpector-main/src/optimizer"],
        library_dirs=["NeuroSpector-main"],
        runtime_library_dirs=["NeuroSpector-main"],
        libraries=["neurospector"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="analyzer_wrapper",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

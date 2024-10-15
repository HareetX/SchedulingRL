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
        [os.path.dirname(os.getcwd())+"/analyzer_wrapper/analyzer_wrapper.cc"],
        include_dirs=[os.path.dirname(os.getcwd()),
                      os.path.dirname(os.getcwd())+"/analyzer",
                      os.path.dirname(os.getcwd())+"/common",
                      os.path.dirname(os.getcwd())+"/utility"],
        library_dirs=[os.path.dirname(os.path.dirname(os.getcwd()))],
        runtime_library_dirs=[os.path.dirname(os.path.dirname(os.getcwd()))],
        libraries=["analyzerwrapper"],
        extra_compile_args=extra_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="analyzer_wrapper",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)

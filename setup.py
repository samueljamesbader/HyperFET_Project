from distutils.core import setup
from Cython.Build import cythonize
import numpy
import os.path
ROOT_DIR = os.path.abspath(os.path.dirname(__file__))

ext_options = {"compiler_directives": {"profile": True}, "annotate": True}

setup(
    name='HyperFET',
    version='0.1.2',
    packages=['hyperfet'],
    url='',
    license='',
    author='Samuel James Bader',
    author_email='samuel.james.bader@gmail.com',
    description='Codebase to accompany the HyperFET project within the Jena-Xing group',
    ext_modules = cythonize([
        "hyperfet/devices.pyx",
        ], **ext_options),
    requires=['numpy', 'matplotlib', 'scipy', 'pytest', 'cython', 'pint'],
    include_dirs=[numpy.get_include(),ROOT_DIR]
)

#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.10.0'
__minimum_tensorflow_version__ = '1.14.0'

setup_requires = ['numpy>=' + __minimum_numpy_version__, 
'tensorflow>='+__minimum_tensorflow_version__]

setup(name='gaia_deprojection',
      version='0.0.1',
      description='Uses AI to probabilistically deproject Gaia observables into star clusters.',
      author=['Josh Albert', 'Santiago Torres', 'Simon Portegies Zwart'],
      author_email=['albert@strw.leidenuniv.nl', 'storres@strw.leidenuniv.nl', 'spz@strw.leidenuniv.nl'],
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_data= {'gaia_deproject':['data/*']},
   package_dir = {'':'./'},
   packages=find_packages('./')
     )


from setuptools import setup

setup(name='multipoles',
      version='0.1',
      description='Creates and manipulates multipole fields',
      url='http://github.com/tobinjones/multipoles/',
      author='Tobin Jones',
      author_email='tobin.jones@auckland.ac.nz',
      license='All rights reserved',
      packages=['multipoles'],
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
      ],
      zip_safe=False)

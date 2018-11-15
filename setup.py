##
# \file setup.py
#
# Instructions:
# 1) Set environment variables with prefix NIFTYMIC_, e.g.
#   `export NIFTYMIC_ITK_DIR=path-to-ITK_NIFTYMIC-build`
#   to incorporate `-D ITK_DIR=path-to-ITK_NIFTYMIC-build` in `cmake` build.
# 2) `pip install -e .`
#   All python packages and command line tools are then installed during
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#


import re
import os
import sys
from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install

from install_cli import main as install_command_line_interfaces


##
# Post-installation to build additionally required command line interface tools
# located in niftymic/cli.
# \date       2017-10-20 17:00:53+0100
#
class CustomDevelopCommand(develop):

    def run(self):
        install_command_line_interfaces()
        develop.run(self)


##
# Post-installation to build additionally required command line interface tools
# located in niftymic/cli.
# \date       2017-10-20 17:00:53+0100
#
class CustomInstallCommand(install):

    def run(self):
        install_command_line_interfaces()
        install.run(self)


description = "Motion Correction and Volumetric Image Reconstruction of 2D " \
    "Ultra-fast MRI"
long_description = "NiftyMIC is a research-focused toolkit developed within the " \
    "[GIFT-Surg](http://www.gift-surg.ac.uk/) project to reconstruct an " \
    "isotropic, high-resolution volume from multiple, possibly " \
    "motion-corrupted, stacks of low-resolution 2D slices. The framework " \
    "relies on slice-to-volume registration algorithms for motion " \
    "correction and reconstruction-based Super-Resolution(SR) techniques " \
    "for the volumetric reconstruction." \
    "The entire reconstruction pipeline is programmed in Python by using a " \
    "mix of SimpleITK, WrapITK and standard C++ITK."


setup(name='NiftyMIC',
      version='0.5rc1',
      description=description,
      long_description=long_description,
      url='https://github.com/gift-surg/NiftyMIC',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['niftymic'],
      install_requires=[
          'pysitk>=0.2',
          'nsol>=0.1.5',
          'simplereg>=0.2',
          'scikit_image>=0.12.3',
          'scipy>=0.19.1',
          'natsort>=5.0.3',
          'numpy>=1.13.1',
          'SimpleITK>=1.0.1',
          'pymc3>=3.3',
          'theano>=1.0.1',
          "six>=1.10.0",
      ],
      zip_safe=False,
      keywords='development numericalsolver convexoptimisation',
      classifiers=[
          'Intended Audience :: Developers',
          'Intended Audience :: Healthcare Industry',
          'Intended Audience :: Science/Research',

          'License :: OSI Approved :: BSD License',

          'Topic :: Software Development :: Build Tools',
          'Topic :: Scientific/Engineering :: Medical Science Apps.',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 3',
      ],
      cmdclass={
          "develop": CustomDevelopCommand,
          "install": CustomInstallCommand,
      },
      entry_points={
          'console_scripts': [
              'niftymic_correct_bias_field = niftymic.application.correct_bias_field:main',
              'niftymic_reconstruct_volume = niftymic.application.reconstruct_volume:main',
              'niftymic_reconstruct_volume_from_slices = niftymic.application.reconstruct_volume_from_slices:main',
              'niftymic_register_image = niftymic.application.register_image:main',
              'niftymic_multiply_stack_with_mask = niftymic.application.multiply_stack_with_mask:main',
              'niftymic_run_reconstruction_parameter_study = niftymic.application.run_reconstruction_parameter_study:main',
              'niftymic_run_reconstruction_pipeline = niftymic.application.run_reconstruction_pipeline:main',
              'niftymic_nifti2dicom = niftymic.application.nifti2dicom:main',
              'niftymic_show_reconstruction_parameter_study = nsol.application.show_parameter_study:main',
          ],
      },
      )

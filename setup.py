###
# \file setup.py
#
# Install with symlink: 'pip install -e .'
# Changes to the source file will be immediately available to other users
# of the package
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       July 2017
#
# \see https://python-packaging.readthedocs.io/en/latest/minimal.html
# \see https://python-packaging-user-guide.readthedocs.io/tutorials/distributing-packages/


from setuptools import setup

description = "Motion correction and volumetric Image ReConstruction of 2D ultra-fast MRI"
long_description = "This is a research-focused toolkit developed within the" \
    " [GIFT-Surg](http: // www.gift-surg.ac.uk/) project to reconstruct an " \
    "isotropic, high-resolution volume from multiple, possibly " \
    "motion-corrupted, stacks of low-resolution 2D slices. The framework " \
    "relies on slice-to-volume registration algorithms for motion correction " \
    "and reconstruction-based Super-Resolution(SR) techniques for the " \
    "volumetric reconstruction." \
    "The entire reconstruction pipeline is programmed in Python by using a " \
    "mix of SimpleITK, WrapITK and standard C++ITK."

setup(name='NiftyMIC',
      version='0.1.dev1',
      description=description,
      long_description=long_description,
      url='https://github.com/gift-surg/NiftyMIC',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='BSD-3-Clause',
      packages=['niftymic'],
      install_requires=[
          'pysitk',
          'nsol',
          'simplereg',
          'scikit_image>=0.12.3',
          'scipy>=0.19.1',
          'natsort>=5.0.3',
          'matplotlib>=2.0.2',
          'numpy>=1.13.1',
          'SimpleITK>=1.0.1',
      ],
      zip_safe=False,
      keywords='development numericalsolver convexoptimisation',
      classifiers=[
          'Development Status :: 3 - Alpha',

          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          'License :: OSI Approved :: BSD License',

          'Programming Language :: Python',
          'Programming Language :: Python :: 2',
          'Programming Language :: Python :: 2.7',
      ],
      entry_points={
          'console_scripts': [
              'niftymic_correct_bias_field = niftymic.application.correct_bias_field:main',
              'niftymic_reconstruct_volume = niftymic.application.reconstruct_volume:main',
              'niftymic_reconstruct_volume_from_slices = niftymic.application.reconstruct_volume_from_slices:main',
              'niftymic_register_to_template = niftymic.application.register_to_template:main',
              'niftymic_run_intensity_correction = niftymic.application.run_intensity_correction:main',
              'niftymic_run_reconstruction_parameter_study = niftymic.application.run_reconstruction_parameter_study:main',
              'niftymic_show_reconstruction_parameter_study = nsol.application.show_parameter_study:main',
          ],
      },
      )

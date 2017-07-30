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

long_description = "This is a research-focused toolkit developed within the" \
    " [GIFT-Surg](http: // www.gift-surg.ac.uk/) project to reconstruct an " \
    "isotropic, high-resolution volume from multiple, possibly " \
    "motion-corrupted, stacks of low-resolution 2D slices. The framework " \
    "relies on slice-to-volume registration algorithms for motion correction " \
    "and reconstruction-based Super-Resolution(SR) techniques for the " \
    "volumetric reconstruction." \
    "The entire reconstruction pipeline is programmed in Python by using a " \
    "mix of SimpleITK, WrapITK and standard C++ITK."

setup(name='VolumetricReconstruction',
      version='0.1.dev1',
      description='Volumetric MRI Reconstruction from 2D Slices in the '
      'Presence of Motion',
      long_description=long_description,
      url='https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction',
      author='Michael Ebner',
      author_email='michael.ebner.14@ucl.ac.uk',
      license='MIT',
      packages=['volumetricreconstruction'],
      package_dir={'': 'src/py'},  # tell distutils packages are under src/py
      # include_package_data=True,    # include everything in source control
      install_requires=['pythonhelper', 'numericalsolver'],
      zip_safe=False,
      keywords='development numericalsolver convexoptimisation',
      classifiers=[
          # How mature is this project? Common values are
          #   3 - Alpha
          #   4 - Beta
          #   5 - Production/Stable
          'Development Status :: 3 - Alpha',

          # Indicate who your project is intended for
          'Intended Audience :: Developers',
          'Topic :: Software Development :: Build Tools',

          # Pick your license as you wish (should match "license" above)
           'License :: OSI Approved :: MIT License',

          # Specify the Python versions you support here. In particular, ensure
          # that you indicate whether you support Python 2, Python 3 or both.
          # 'Programming Language :: Python :: 2',
          # 'Programming Language :: Python :: 2.6',
          'Programming Language :: Python :: 2.7',
          # 'Programming Language :: Python :: 3',
          # 'Programming Language :: Python :: 3.2',
          # 'Programming Language :: Python :: 3.3',
          # 'Programming Language :: Python :: 3.4',
      ],

      )

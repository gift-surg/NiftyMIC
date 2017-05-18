# Volumetric MRI Reconstruction from Motion Corrupted 2D Slices

This toolkit is a research-focussed tool developed within the [GIFT-Surg](http://www.gift-surg.ac.uk/) project to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2D slices. The framework relies on slice-to-volume registration algorithms for motion correction and reconstruction-based Super-Resolution (SR) techniques for the volumetric reconstruction. 
The entire reconstruction pipeline is programmed in Python by using a mix of SimpleITK, WrapITK and standard C++ITK. Several functions are added to the
standard ITK package and wrapped so that they are available in Python.

This toolkit is still under development and has NOT been publicly released yet. In case you have access to this code, please do not share it without approval.

If you have any questions or comments (or find bugs), please drop an email to @mebner (`michael.ebner.14@ucl.ac.uk`).

# Installation
Clone the Volumetric MRI Reconstruction Toolkit by
* `git clone git@cmiclab.cs.ucl.ac.uk:mebner/VolumetricReconstruction.git`

enter the root directory (`VolumetricReconstruction`) and change to the `dev` branch, i.e.

* `cd VolumetricReconstruction`
* `git checkout dev`

## Installation of Python packages
The required Python packages are stated in the file `src/py/requirements.txt`. They can be installed manually or by running
* `pip install -r src/py/requirements.txt`

## Installation of (Wrap)ITK
Installation of ITK and its wrapping to Python is the most time-consuming process. The documentation on how-to can be found [here](https://cmiclab.cs.ucl.ac.uk/mebner/ITK/wikis/home).

## Optional Packages

### Build ITK-cpp code
In case you want to use the classes
* `N4BiasFieldCorrection`
* `RegistrationITK`
* `NiftyReg`
* `SIENA`
* `BrainStripping`

(and possibly others) you will need to compile the code in `src/cpp`. For doing so, compile the source code by linking it with the built ITK libraries, via
* `mkdir -p build/cpp/`
* `cd build/cpp/`
* `cmake -D ITK_DIR=path-to-ITK-build_dev ../../src/cpp/`
* `make -j`

### NiftyReg
The class `NiftyReg` provides a basic wrapper for [NiftyReg](http://cmictig.cs.ucl.ac.uk/wiki/index.php/NiftyReg) and makes both `reg_aladin` and `reg_f3d` accessible to Python. Installation instructions for NiftyReg can be found on the website.

### Visualisation
The Volumetric MRI Reconstruction Toolkit uses [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) for the visualization of images. In case you want to make use of it, make sure ITK-SNAP is installed and can be accessed via `itksnap` from the command line.

## Code Documentation
A documentation for the Python source-files can be generated in case [Doxygen](http://www.doxygen.org) is installed. Within the root folder `VolumetricReconstruction` run
* `cd doc/py`
* `doxygen doxyfile`
* `open html/index.html`

# Example usage
A simple example (without motion correction) can be found in `src/py/reconstructStaticVolume`. With the test data from [here](https://www.dropbox.com/sh/je6luff8y8d692e/AABx798T_PyaIXXsh0pq7rVca?dl=0) the reconstruction can be run by
* `cd src/py`
* `python reconstructStaticVolume.py --dir_input=path-to-fetal-data --dir_output=path-to-output-dir --target_stack_index=1`

# References
Associated publications are 
* [[Ebner2017]](https://link.springer.com/chapter/10.1007%2F978-3-319-52280-7_1) Ebner, M., Chouhan, M., Patel, P. A., Atkinson, D., Amin, Z., Read, S., Punwani, S., Taylor, S., Vercauteren, T., and Ourselin, S. (2017). Point-Spread-Function-Aware Slice-to-Volume Registration: Application to Upper Abdominal MRI Super-Resolution. In Zuluaga, M. A., Bhatia, K., Kainz, B., Moghari, M. H., and Pace, D. F., editors, Reconstruction, Segmentation, and Analysis of Medical Images. RAMBO 2016, volume 10129 of Lecture Notes in Computer Science, pages 3–13. Springer International Publishing.
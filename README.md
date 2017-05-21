# Volumetric MRI Reconstruction from Motion Corrupted 2D Slices

This toolkit is a research-focussed tool developed within the [GIFT-Surg](http://www.gift-surg.ac.uk/) project to reconstruct an isotropic, high-resolution volume from multiple, possibly motion-corrupted, stacks of low-resolution 2D slices. The framework relies on slice-to-volume registration algorithms for motion correction and reconstruction-based Super-Resolution (SR) techniques for the volumetric reconstruction. 
The entire reconstruction pipeline is programmed in Python by using a mix of SimpleITK, WrapITK and standard C++ITK. Several functions are added to the
standard ITK package and wrapped so that they are available in Python.

If you have any questions or comments (or find bugs), please drop an email to @mebner (`michael.ebner.14@ucl.ac.uk`).


# Installation
Clone the Volumetric MRI Reconstruction Toolkit by
* `git clone git@cmiclab.cs.ucl.ac.uk:mebner/VolumetricReconstruction.git`

and add this path to the environment variable `VOLUMETRIC_RECONSTRUCTION_DIR` 
by adding the line
* `export VOLUMETRIC_RECONSTRUCTION_DIR="path-to-VolumetricReconstrucion/"`

in your `.bashrc` file (or whatever you use for your terminal). Then the 
Volumetric Reconstruction Toolkit modules 
can be included in Python via `sys.path.insert(1, os.path.abspath(os.path.join(os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py')))` later on.

## Installation of additional Python packages
This toolkit was developed and tested with Python 2 and relies on additional Python packages.
The required packages (and their tested versions) are stated in the file `requirements.txt`. These are 
* `numpy`
* `scipy`
* `matplotlib`
* `SimpleITK`
* `nibabel`
* `Pillow`

and can be installed with `pip` (version >= 9.0) by running the command
* `pip install -r requirements.txt`


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

### Visualization
The Volumetric MRI Reconstruction Toolkit uses [ITK-SNAP](http://www.itksnap.org/pmwiki/pmwiki.php) for the visualization of images. In case you want to make use of it, make sure ITK-SNAP is installed and can be accessed via `itksnap` from the command line.

## Code Documentation
A documentation for the Python source-files can be generated in case [Doxygen](http://www.doxygen.org) is installed. Within the root folder `VolumetricReconstruction` run
* `cd doc/py`
* `doxygen doxyfile`
* `open html/index.html`


# Example usage
A simple example (without motion correction) can be found in `examples/reconstructStaticVolume`. Test data including three orthogonal acquisitions of the fetal brain can be downloaded by either using this [link](https://www.dropbox.com/sh/je6luff8y8d692e/AABx798T_PyaIXXsh0pq7rVca?dl=0) or executing
* `curl -L https://www.dropbox.com/sh/je6luff8y8d692e/AABx798T_PyaIXXsh0pq7rVca?dl=1 > fetal_brain.zip`
* `unzip fetal_brain.zip -d fetal_brain`

in the terminal. The Super-Resolution reconstruction algorithm can be run by
* `python examples/reconstructStaticVolume.py --dir_input=fetal_brain --dir_output=results --target_stack_index=1`

# References
Associated publications are 
* [[Ebner2017]](https://link.springer.com/chapter/10.1007%2F978-3-319-52280-7_1) Ebner, M., Chouhan, M., Patel, P. A., Atkinson, D., Amin, Z., Read, S., Punwani, S., Taylor, S., Vercauteren, T., and Ourselin, S. (2017). Point-Spread-Function-Aware Slice-to-Volume Registration: Application to Upper Abdominal MRI Super-Resolution. In Zuluaga, M. A., Bhatia, K., Kainz, B., Moghari, M. H., and Pace, D. F., editors, Reconstruction, Segmentation, and Analysis of Medical Images. RAMBO 2016, volume 10129 of Lecture Notes in Computer Science, pages 3–13. Springer International Publishing.

# License
This toolkit is still under development and has NOT been publicly released yet.
See LICENSE file for details.

# Volumetric MRI Reconstruction from Motion Corrupted 2D Slices

The entire reconstruction pipeline was programmed in Python by using a mix 
of SimpleITK, WrapITK and standard C++ITK. Several functions were added to the
standard ITK package and wrapped so that they are made available within Python.

Hence, in order to use this package, several libraries need to be installed.
For further information have a look at the 
[Wiki](https://cmiclab.cs.ucl.ac.uk/mebner/VolumetricReconstruction/wikis/home).
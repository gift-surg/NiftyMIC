#!/usr/bin/python

##
# \file run_tests.py
# \brief      main-file to run specified unit tests
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2015
#


# Import libraries
import unittest
import sys
import os

# Import modules for unit testing
from brain_stripping_test import *
# from cpp_itk_registration_test import *
# from differential_operations_test import *
# from intensity_correction_test import *
# from intra_stack_registration_test import *
# from linear_image_quality_transfer_test import *
# from niftyreg_test import *
# from parameter_normalization_test import *
# from registration_test import *
# from segmentation_propagation_test import *
# from simulator_slice_acquisition_test import *
# from stack_test import *


if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()

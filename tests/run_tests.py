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
from modules.brain_stripping_test import *
from modules.cpp_itk_registration_test import *
from modules.differential_operations_test import *
from modules.intensity_correction_test import *
from modules.intra_stack_registration_test import *
from modules.linear_image_quality_transfer_test import *
from modules.niftyreg_test import *
from modules.parameter_normalization_test import *
from modules.registration_test import *
from modules.segmentation_propagation_test import *
from modules.simulator_slice_acquisition_test import *
from modules.stack_test import *


if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()

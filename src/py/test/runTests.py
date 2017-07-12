#!/usr/bin/python

##
# \file runTests.py
# \brief      main-file to run specified unit tests
#
# \author     Michael Ebner (michael.ebner.14@ucl.ac.uk)
# \date       September 2015
#


# Import libraries
import unittest
import sys
import os

# Add directories to import modules
sys.path.insert(1, os.path.abspath(os.path.join(
    os.environ['VOLUMETRIC_RECONSTRUCTION_DIR'], 'src', 'py')))

# Import modules for unit testing
# from modules.TestFirstEstimateOfHRVolume import * ## outdated
# from modules.TestHierarchicalSliceAlignment import * ## outdated
# from modules.TestInPlaneRegistrationSimpleITK import * ## outdated
from modules.TestSimpleITKHelper import *
from modules.TestStack import *
from modules.TestSimulatorSliceAcqusition import *
from modules.TestNiftyReg import *
from modules.TestDifferentialOperations import *
from modules.TestRegistration import *
from modules.TestBrainStripping import *
from modules.TestIntraStackRegistration import *
from modules.TestParameterNormalization import *
from modules.TestRegistrationITK import *
from modules.TestLinearImageQualityTransfer import *
from modules.TestIntensityCorrection import *
from modules.TestSegmentationPropagation import *
from modules.TestLossFunctions import *

""" ###########################################################################
Main Function
"""
if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()

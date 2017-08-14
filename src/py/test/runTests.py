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

# Import modules for unit testing
# from modules.TestFirstEstimateOfHRVolume import * ## outdated
# from modules.TestHierarchicalSliceAlignment import * ## outdated
# from modules.TestInPlaneRegistrationSimpleITK import * ## outdated

# from modules.TestStack import *
# from modules.TestSimulatorSliceAcqusition import *
# from modules.TestNiftyReg import *
# from modules.TestDifferentialOperations import *
from modules.TestRegistration import *
# from modules.TestBrainStripping import *
# from modules.TestIntraStackRegistration import *
# from modules.TestParameterNormalization import *
# from modules.TestRegistrationCppITK import *
# from modules.TestLinearImageQualityTransfer import *
# from modules.TestIntensityCorrection import *
# from modules.TestSegmentationPropagation import *

""" ###########################################################################
Main Function
"""
if __name__ == '__main__':
    print("\nUnit tests:\n--------------")
    unittest.main()

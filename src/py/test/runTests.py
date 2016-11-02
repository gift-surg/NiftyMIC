#!/usr/bin/python

## \file runTests.py
#  \brief main-file incorporating all the other files 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015



## Import libraries 
import unittest
import sys

# sys.modules.clear()

## Add directories to import modules
dir_src_root = "../"
sys.path.append( dir_src_root )

## Import modules for unit testing
# from modules.TestFirstEstimateOfHRVolume import *
# from modules.TestSimpleITKHelper import *
# from modules.TestStack import *
# from modules.TestSimulatorSliceAcqusition import *
# from modules.TestNiftyReg import *
# from modules.TestDifferentialOperations import *
# from modules.TestRegistration import *
# from modules.TestBrainStripping import *
from modules.TestStackInPlaneAlignment import *
# from modules.TestRegistrationITK import *


""" ###########################################################################
Main Function
"""
if __name__ == '__main__':
    """
    Choose variables
    """
    ## Directory to save obtained results
    # dir_output = "results/"

    ## Choose decimal place accuracy for unit tests:
    # accuracy = 6

    """
    Unit tests:
    """
    ## Prepare output directory
    # reconstruction_manager = rm.ReconstructionManager(dir_output)

    # ## Read input data
    # dir_input, filenames = read_input_data(input_stacks_type)

    # reconstruction_manager.read_input_data(dir_input, filenames)

    
    print("\nUnit tests:\n--------------")
    unittest.main()

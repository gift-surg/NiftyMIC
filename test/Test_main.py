#!/usr/bin/python

## \file Test_main.py
#  \brief main-file incorporating all the other files 
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)#  \date September 2015


## Import libraries 
import SimpleITK as sitk
# import numpy as np
import unittest

import sys
sys.path.append("modules/")

import matplotlib.pyplot as plt

## Import modules from src-folder
# import ReconstructionManager as rm
# import SimpleITKHelper as sitkh
# import InPlaneRigidRegistration as inplaneRR

## Import modules for unit testing
# from Test_FirstEstimateOfHRVolume import *
# from Test_SimpleITKHelper import *
# from Test_Stack import *
from Test_SliceAcqusition import *


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

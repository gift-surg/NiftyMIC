#!/usr/bin/python

## \file reconstructVolume.py
#  \brief  Reconstruction of fetal brain.
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date March 2016

## Import libraries 
import SimpleITK as sitk
import itk

import numpy as np
from scipy.optimize import minimize
from scipy.optimize import leastsq
from scipy import ndimage
import unittest
import matplotlib.pyplot as plt
import sys
import time

## Import modules
import base.Stack as st
import utilities.PythonHelper as ph
import utilities.SimpleITKHelper as sitkh

from definitions import dir_test


"""
Main Function
"""
if __name__ == '__main__':

    # image_original_sitk = sitk.ReadImage("/Users/mebner/A0480244-B6254806-1yr-0.dcm")
    # image_sitk = sitkh.get_downsampled_sitk_image(image_original_sitk,(10,10,1))

    trafo_sitk = sitk.Euler3DTransform()
    trafo_sitk.SetRotation(0,0,-np.pi/2)
    matrix_sitk = trafo_sitk.GetMatrix()

    image_sitk.SetDirection(matrix_sitk)
    sitkh.show_sitk_image(image_sitk)

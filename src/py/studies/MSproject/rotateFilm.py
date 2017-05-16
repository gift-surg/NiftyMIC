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

    dir_input = "/Users/mebner/"
    dir_output = "/tmp/"
    filename = "A0480244-B6254806-1yr-0_old"

    flag_test = 1
    downsampling = 10

    image_sitk = sitk.ReadImage(dir_input + filename + ".dcm")

    if flag_test:
        ## For faster processing/checking/testing
        image_sitk = sitkh.get_downsampled_sitk_image(image_sitk,(downsampling,downsampling,1))

    ## Variant A: Change image header does not work. Storing the file as nifti
    ## with updated direction does not work -> Change data array, i.e. Variant B
    # trafo_sitk = sitk.Euler3DTransform()
    # trafo_sitk.SetRotation(0,0,-np.pi/2)
    # matrix_sitk = trafo_sitk.GetMatrix()
    # matrix_sitk = np.round(matrix_sitk)
    # image_sitk.SetDirection(matrix_sitk)

    ## Variant B: Reshape data array:
    nda = sitk.GetArrayFromImage(image_sitk)
    shape = nda.shape
    nda_rotated = np.zeros((shape[0], shape[2], shape[1]))
    for i in xrange(0, shape[2]):
        nda_rotated[0,i,:] = nda[0,:,-i]
    image_sitk = sitk.GetImageFromArray(nda_rotated)

    if flag_test:
        sitkh.show_sitk_image(image_sitk)
    
    ## It does not work to write the image (correctly) to a DICOM image
    ## Workaround: Open nifti in XMedCon and save it to DICOM manually
    # else:
        sitk.WriteImage(image_sitk, dir_output + filename + ".nii.gz")
        
        ph.execute_command("xmedcon -f " + dir_output + filename + ".nii.gz & ")

    # print image_original_sitk.GetDirection()
    # image_original_sitk.SetDirection(matrix_sitk)
    # # print image_original_sitk.GetDirection()
    
    # # sitkh.show_sitk_image(image_original_sitk)
    # sitk.WriteImage(image_original_sitk, dir_output + filename + ".nii.gz")


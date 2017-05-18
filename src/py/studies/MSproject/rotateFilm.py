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

def get_90_degree_rotated_image(image_sitk):
    nda = sitk.GetArrayFromImage(image_sitk)
    shape = nda.shape
    nda_rotated = np.zeros((shape[0], shape[2], shape[1]))
    for i in xrange(0, shape[2]):
        nda_rotated[0,i,:] = nda[0,:,-i]
    image_sitk = sitk.GetImageFromArray(nda_rotated)

    return image_sitk

def get_180_degree_rotated_image(image_sitk):
    
    image_sitk = get_90_degree_rotated_image(image_sitk)
    image_sitk = get_90_degree_rotated_image(image_sitk)

    return image_sitk



"""
Main Function
"""
if __name__ == '__main__':

    dir_input = "/Volumes/UCLMEBNER1TB/data_for_michael/"
    dir_subfolder = "1yr_3x3/"
    filename = "A8999863-B1178416-1yr-1"
    
    dir_output = "/tmp/"

    flag_test = 0
    downsampling = 5

    image_sitk = sitk.ReadImage(dir_input + dir_subfolder + filename + ".dcm")

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
    image_sitk = get_180_degree_rotated_image(image_sitk)

    if flag_test:
        sitkh.show_sitk_image(image_sitk, label=filename)
    
    ## It does not work to write the image (correctly) to a DICOM image
    ## Workaround: Open nifti in XMedCon and save it to DICOM manually
    else:
        sitk.WriteImage(image_sitk, dir_output + filename + ".nii.gz")
        
        ph.execute_command("xmedcon -f " + dir_output + filename + ".nii.gz & ")


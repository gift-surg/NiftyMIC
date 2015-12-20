#!/usr/bin/python

## \file SimpleITK_ScatteredDataApproximation.py
#  \brief Figure out how to implement and use Scattered Data Reconstruction methods in SimpleITK:
#       - Scattered Data Approximation (SDA)
#
#  \author: Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date: Octoboer 2015

## Import libraries
import SimpleITK as sitk
# import nibabel as nib
import numpy as np
import unittest

import os                       # used to execute terminal commands in python
import sys
sys.path.append("../")

## Import modules from src-folder
# import SimpleITKHelper as sitkh


"""
Functions
"""
def permute_index(shape, i, j, k):
    z_axis = np.argmin(shape)

    index = range(0,len(shape))

    in_plane = np.delete(index, z_axis)
    x_axis = in_plane[0]
    y_axis = in_plane[1]

    index[x_axis] = i
    index[y_axis] = j
    index[z_axis] = k

    return tuple(index)


def generate_undersampled_image_and_mask(image_sitk, delete_fraction):

    fixed_nda = sitk.GetArrayFromImage(image_sitk)
    shape = fixed_nda.shape

    mask_nda = np.zeros(shape)

    z_axis = np.argmin(shape)
    shape_in_plane = np.delete(shape, z_axis)

    N_random = int(np.round(np.prod(shape_in_plane)*float(delete_fraction)))

    for i in range(0,shape[z_axis]):   
        a = np.random.randint(0, shape_in_plane[0], (2, N_random))
        print("Delete %r %%, i.e. %r out of %r pixels, of slice %r/%r" 
            %(delete_fraction*100, N_random, np.prod(shape_in_plane), i, shape[z_axis]-1))

        for j in range(0,N_random):
            fixed_nda[permute_index(shape, a[0], a[1], i)] = 0
            mask_nda[permute_index(shape, a[0], a[1], i)] = 1

    ## Image
    image_undersampled = sitk.GetImageFromArray(fixed_nda)
    image_undersampled.CopyInformation(image_sitk)

    ## Mask
    mask = sitk.GetImageFromArray(mask_nda)
    mask.CopyInformation(image_sitk)

    return image_undersampled, mask


"""
Unit Test Class
"""

class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_01(self):
        pass


"""
Main Function
"""
if __name__ == '__main__':

    """
    Set variables
    """
    ## Specify data
    dir_input = "data/"
    dir_output = "results/"
    filename =  "CTL_0_baseline"

    accuracy = 6 # decimal places for accuracy of unit tests

    """
    Unit tests:
    """
    # print("\nUnit tests:\n--------------")
    # unittest.main()


    fixed = sitk.ReadImage(dir_input + filename + ".nii.gz", sitk.sitkFloat64)


    # for delete_fraction in [0.8, 0.9]:

    #     fixed_undersampled, mask = generate_undersampled_image_and_mask(fixed, delete_fraction)

    #     sitk.WriteImage(fixed_undersampled, dir_input + filename + "_deleted_" + str(delete_fraction) + ".nii.gz")
    #     sitk.WriteImage(mask, dir_input + filename + "_deleted_" + str(delete_fraction) + "_mask.nii.gz")

    
    delete_fraction = 0.1
    sigma = 0.5

    image = sitk.ReadImage(dir_input + filename + "_deleted_" + str(delete_fraction) + ".nii.gz")

    image_nda = sitk.GetArrayFromImage(image)
    image_nda[:] = 1

    D = sitk.GetImageFromArray(image_nda)
    D.CopyInformation(image)

    gaussian = sitk.SmoothingRecursiveGaussianImageFilter()
    gaussian.SetSigma(sigma)

    N = gaussian.Execute(image)
    D = gaussian.Execute(D)

    output = N/D

    sitk.WriteImage(output, dir_output + filename+ "_deleted_" + str(delete_fraction) + "_reconstructed_sigma_" + str(sigma) +".nii.gz")



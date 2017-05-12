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

PIXEL_TYPE = itk.D
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]
IMAGE_TYPE_CV33 = itk.Image.CVD33
IMAGE_TYPE_CV183 = itk.Image.CVD183

"""
Main Function
"""
if __name__ == '__main__':

    itk2np = itk.PyBuffer[IMAGE_TYPE]
    itk2np_CVD33 = itk.PyBuffer[IMAGE_TYPE_CV33]
    itk2np_CVD183 = itk.PyBuffer[IMAGE_TYPE_CV183]

    filename = "FetalBrain_reconstruction_3stacks_myAlg"
    HR_volume = st.Stack.from_filename(dir_test, filename)
    
    DOF_transform = 6
    parameters = np.random.rand(DOF_transform)*(2*np.pi, 2*np.pi, 2*np.pi, 10, 10, 10)

    transform_itk = itk.Euler3DTransform.New()
    parameters_itk = transform_itk.GetParameters()
    sitkh.update_itk_parameters(parameters_itk, parameters)
    transform_itk.SetParameters(parameters_itk)

    ##---------------------------------------------------------------------
    time_start = ph.start_timing()
    filter_gradient_transform = itk.GradientEuler3DTransformImageFilter[IMAGE_TYPE, PIXEL_TYPE, PIXEL_TYPE].New()
    filter_gradient_transform.SetInput(HR_volume.itk)
    filter_gradient_transform.SetTransform(transform_itk)
    filter_gradient_transform.Update()
    gradient_transform_itk = filter_gradient_transform.GetOutput()
    ## Get data array of Jacobian of transform w.r.t. parameters  and 
    ## reshape to N_HR_volume_voxels x DIMENSION x DOF
    nda_gradient_transform_1 = itk2np_CVD183.GetArrayFromImage(gradient_transform_itk).reshape(-1,3,DOF_transform)
    
    print("GradientEuler3DTransformImageFilter: " + str(ph.stop_timing(time_start)))

    ##-------------------------------------------------------------------------
    image_sitk = HR_volume.sitk

    time_start = ph.start_timing()
    shape = np.array(image_sitk.GetSize())
    dim = image_sitk.GetDimension()

    ## Index array (dimension x N_voxels) of image in voxel space
    indices = sitkh.get_indices_array_to_flattened_sitk_image_data_array(image_sitk)
    
    ## Get transform from voxel to image space coordinates
    A = sitkh.get_sitk_affine_matrix_from_sitk_image(image_sitk).reshape(dim,dim)
    t = np.array(image_sitk.GetOrigin()).reshape(dim,1)
    
    ## Compute point array (3xN_voxels) of image in image space
    points = A.dot(indices) + t

    ## Allocate memory
    transform_dof = int(transform_itk.GetNumberOfParameters())
    jacobian_transform_on_image_nda = np.zeros((points.shape[1], dim, transform_dof))

    ## Create 2D itk-array
    jacobian_transform_on_point_itk = itk.Array2D[itk.D]()
    
    ## Evaluate the Jacobian of transform at all points
    for i in range(0, points.shape[1]):
        
        ## Compute Jacobian of transform w.r.t. parameters evaluated at point
        ## jacobian_transform_point_itk is (Dimension x transform_DOF) array
        # time_start = ph.start_timing()
        transform_itk.ComputeJacobianWithRespectToParameters(points[:,i], jacobian_transform_on_point_itk)
        # print("ComputeJacobianWithRespectToParameters: " + str(ph.stop_timing(time_start)))

        ## Convert itk to numpy array
        ## THE computational time consuming part!
        # print jacobian_transform_on_image_nda

        jacobian_transform_on_image_nda[i,:,:] = sitkh.get_numpy_from_itk_array(jacobian_transform_on_point_itk)

    print("get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image: " + str(ph.stop_timing(time_start)))

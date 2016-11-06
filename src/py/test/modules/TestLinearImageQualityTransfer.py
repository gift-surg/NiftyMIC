## \file TestLinearImageQualityTransfer.py
#  \brief  Class containing unit tests for module DifferentialOperations
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date July 2016


## Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest
import sys
from scipy import ndimage

## Import modules from src-folder
import reconstruction.LinearImageQualityTransfer as liqt


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestLinearImageQualityTransfer(unittest.TestCase):

    ## Specify input data
    dir_test_data = "../../../test-data/"

    accuracy = 6

    # def setUp(self):
    #     pass

    def test_kernel_2D_as_kernel_3D(self):

        N = 6

        nda_shape = (40,200,200)
        kernel = np.arange(1, N*N+1)

        kernel_2D = kernel.reshape(N,N)
        kernel_3D = kernel.reshape(1,N,N)

        nda =  255 * np.random.rand(nda_shape[0],nda_shape[1],nda_shape[2])

        nda_2D = np.array(nda)
        nda_3D = np.array(nda)

        ## 2D
        for i in range(0, nda.shape[0]):
            nda_2D[i,:,:] = ndimage.convolve(nda_2D[i,:,:], kernel_2D)

        ## 3D
        nda_3D = ndimage.convolve(nda_3D, kernel_3D)

        self.assertEqual(np.around(
            np.linalg.norm(nda_2D-nda_3D)
            , decimals = self.accuracy), 0 )
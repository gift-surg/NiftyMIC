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

    def setUp(self):
        pass

    ##
    #       Test whether interpretation of a 2D kernel in 3D is correct
    # \date       2016-11-06 15:26:19+0000
    #
    # \param      self  The object
    #
    def test_kernel_2D_as_kernel_3D(self):

        ## Shape in (z,y,x)-coordinates
        nda_shape = (40,200,200)
        
        ## Define size of kernel
        N = 6

        ## Create random data array
        nda =  255 * np.random.rand(nda_shape[0],nda_shape[1],nda_shape[2])

        ## Create kernel with elements 1:N^2
        kernel = np.arange(1, N*N+1)

        ## Define 2D- and equivalent 3D-kernel
        kernel_2D = kernel.reshape(N,N)
        kernel_3D = kernel.reshape(1,N,N)

        ## Create data array copies
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
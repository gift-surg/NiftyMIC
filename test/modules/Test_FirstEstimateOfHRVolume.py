## \file Test_FirstEstimateOfHRVolume.py
#  \brief  Class containing unit tests for module FirstEstimateOfHRVolume
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


# Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest

import sys
sys.path.append("../src/")
sys.path.append("data/FirstEstimateOfHRVolume/")

## Import modules from src-folder
import FirstEstimateOfHRVolume as tm
import StackManager as sm
import Slice as sl
import SimpleITKHelper as sitkh



## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Test_FirstEstimateOfHRVolume(unittest.TestCase):

    ## Specify input data
    dir_input = "data/FirstEstimateOfHRVolume/"
    filenames = ["stack0"]

    accuracy = 8

    def setUp(self):
        pass

    def test_01_update_slice_transformations(self):

        ## Create stack manager and read stack(s)
        stack_manager = sm.StackManager()
        stack_manager.read_input_data(self.dir_input, self.filenames)

        ##
        i = 0
        stack = stack_manager._stacks[i]
        slices_sitk = [None]*stack.get_number_of_slices()

        ## Update slices according to given test data
        for j in range(0, stack.get_number_of_slices()):
            slices_sitk[j] = sitk.ReadImage(self.dir_input + self.filenames[i] + "_" + str(j) + ".nii.gz")
            stack._slices[j] = sl.Slice(sitk.Image(slices_sitk[j]), self.dir_input, self.filenames[i], j)

        ## Fetch rigid registration
        rigid_registrations = [None]*stack_manager.get_number_of_stacks()
        # rigid_registrations[i] = sitk.Euler3DTransform(sitk.ReadTransform(self.dir_input + "rigid_transform_minorDiff.tfm"))
        rigid_registrations[i] = sitk.Euler3DTransform(sitk.ReadTransform(self.dir_input + "rigid_transform_majorDiff.tfm"))

        ## Insantiate testmodule
        testmod = tm.FirstEstimateOfHRVolume(stack_manager, "foo")

        ## Run updates for slice trafos
        testmod._update_slice_transformations(rigid_registrations)

        ## Define vectors pointing along the main axis of the image space
        N_x, N_y, N_z = stack.sitk.GetSize()
        
        e_0 = (0,0,0)
        e_x = (N_x,0,0)

        ## Compute stack origins
        origin_0 = np.array(slices_sitk[0].GetOrigin())
        origin_0_warped = np.array(stack_manager._stacks[i]._slices[0].sitk.GetOrigin())

        ## Compute direction vector of stack (x-direction in voxel space)
        a_0 = np.array(slices_sitk[0].TransformIndexToPhysicalPoint(e_x)) - origin_0
        a_0_warped = np.array(stack_manager._stacks[i]._slices[0].sitk.TransformIndexToPhysicalPoint(e_x)) - origin_0_warped
        
        for j in xrange(0,stack.get_number_of_slices()):
            ## Compute slice origins
            origin_j = np.array(slices_sitk[j].GetOrigin())
            a_j = np.array(slices_sitk[j].TransformIndexToPhysicalPoint(e_x)) - origin_j

            ## Compute direction vector of slices (x-direction in voxel space)
            origin_j_warped = np.array(stack_manager._stacks[i]._slices[j].sitk.GetOrigin())
            a_j_warped = np.array(stack_manager._stacks[i]._slices[j].sitk.TransformIndexToPhysicalPoint(e_x)) - origin_j_warped

            ## Check that translations relativ to stack origins are identical
            self.assertEqual(np.around(
                abs(np.linalg.norm(origin_j-origin_0) - np.linalg.norm(origin_j_warped-origin_0_warped))
                , decimals = self.accuracy), 0 )

            ## Check that relative rotation is identical
            self.assertEqual(np.around(abs(
                np.dot(a_j,a_0)/(np.linalg.norm(a_j)*np.linalg.norm(a_0)) 
                - np.dot(a_j_warped,a_0_warped)/(np.linalg.norm(a_j_warped)*np.linalg.norm(a_0_warped)))
                , decimals = self.accuracy), 0 )



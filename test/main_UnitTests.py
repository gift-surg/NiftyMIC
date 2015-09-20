#!/usr/bin/python

## \file main.py
#  \brief main-file incorporating all the other files 
# 
#  \author Michael Ebner
#  \date September 2015


## Import libraries 
import SimpleITK as sitk
import numpy as np
import unittest

import sys
sys.path.append("../src/")

## Import modules from src-folder
import ReconstructionManager as rm
# import SliceStack


def read_input_data(image_type):
    
    if image_type in ["fetal_neck"]:
        ## Fetal Neck Images:
        dir_input = "../data/fetal_neck/"
        filenames = [
            "20150115_161038s006a1001_crop",
            "20150115_161038s003a1001_crop",
            "20150115_161038s004a1001_crop",
            "20150115_161038s005a1001_crop",
            "20150115_161038s007a1001_crop",
            "20150115_161038s5005a1001_crop",
            "20150115_161038s5006a1001_crop",
            "20150115_161038s5007a1001_crop"
            ]

    elif image_type in ["kidney"]:
        ## Kidney Images:
        dir_input = "/Users/mebner/UCL/Data/Kidney\\ \\(3T,\\ Philips,\\ UCH,\\ 20150713\\)/Nifti/"
        filenames = [
            "20150713_09583130x3mmlongSENSEs2801a1028",
            "20150713_09583130x3mmlongSENSEs2701a1027",
            "20150713_09583130x3mmlongSENSEs2601a1026",
            "20150713_09583130x3mmlongSENSEs2501a1025",
            "20150713_09583130x3mmlongSENSEs2401a1024",
            "20150713_09583130x3mmlongSENSEs2301a1023"
            ]

    else:
        ## Fetal Neck Images:
        dir_input = "../data/placenta_in-plane_Guotai/"
        filenames = [
            "a13_15"
            ]

    return dir_input, filenames




## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestUM(unittest.TestCase):
 
    def setUp(self):
        pass
 
    def test_01_test_in_plane_registration(self):
        stacks = reconstruction_manager.get_stacks()

        N_stacks = len(stacks)

        Nx,Ny,Nz = stacks[0].sitk.GetSize()
        e_0 = (0,0,0)
        e_x = (Nx,0,0)
        e_y = (0,Ny,0)

        for i in range(0,N_stacks):
            stack = stacks[i]
            slices = stack.get_slices()

            N_slices = stack.sitk.GetDepth()

            for j in range(0,N_slices):
                slice_3D_original = stack.sitk[:,:,j:j+1]
                slice_3D_aligned = slices[j].sitk


                a_0 = np.array(slice_3D_original.TransformIndexToPhysicalPoint(e_0))
                a_x = np.array(slice_3D_original.TransformIndexToPhysicalPoint(e_x)) - a_0
                a_y = np.array(slice_3D_original.TransformIndexToPhysicalPoint(e_y)) - a_0
                a_z = np.cross(a_x,a_y)

                b_0 = np.array(slice_3D_aligned.TransformIndexToPhysicalPoint(e_0))
                b_x = np.array(slice_3D_aligned.TransformIndexToPhysicalPoint(e_x)) - b_0
                b_y = np.array(slice_3D_aligned.TransformIndexToPhysicalPoint(e_y)) - b_0

                t_3D = b_0 - a_0

                print("slice %r: ||t_3D|| = %r" % (j, np.linalg.norm(t_3D)))

                # Check: a_x-a_y-plane orthogonal to b_x-b_y-plane
                self.assertEqual(np.around(
                    abs(a_z.dot(b_x))
                    , decimals = accuracy), 0 )
                self.assertEqual(np.around(
                    abs(a_z.dot(b_y))
                    , decimals = accuracy), 0 )

                # Check: In-plane translation vector (i.e. translation in a_x-b_x-plane)
                self.assertEqual(np.around(
                    abs(a_z.dot(t_3D))
                    , decimals = accuracy), 0 )

                # Check: Isometric transformation:
                self.assertEqual(np.around(
                    abs(np.linalg.norm(a_x) - np.linalg.norm(b_x))
                    , decimals = accuracy), 0 )
                self.assertEqual(np.around(
                    abs(np.linalg.norm(a_y) - np.linalg.norm(b_y))
                    , decimals = accuracy), 0 )
 

                # Check: intensity arrays of both original and aligned stack are equal
                slice_3D_original_nda = sitk.GetArrayFromImage(slice_3D_original)
                slice_3D_aligned_nda = sitk.GetArrayFromImage(slice_3D_aligned)

                self.assertEqual(np.around(
                    np.linalg.norm(slice_3D_original_nda - slice_3D_aligned_nda)
                    , decimals = accuracy), 0 )


""" ###########################################################################
Main Function
"""
if __name__ == '__main__':
    """
    Choose variables
    """
    ## Types of input images to process
    input_stack_types_available = ("fetal_neck", "kidney", "placenta")

    ## Directory to save obtained results
    dir_output = "../results/"

    ## Choose input stacks and reference stack therein
    input_stacks_type = input_stack_types_available[2]
    reference_stack_id = 0

    ## Choose decimal place accuracy for unit tests:
    accuracy = 6

    """
    Unit tests:
    """
    ## Prepare output directory
    reconstruction_manager = rm.ReconstructionManager(dir_output)

    ## Read input data
    dir_input, filenames = read_input_data(input_stacks_type)
    reconstruction_manager.read_input_data(dir_input, filenames)

    ## In-plane rigid registration
    reconstruction_manager.run_in_plane_rigid_registration()

    print("\nUnit tests:\n--------------")
    unittest.main()
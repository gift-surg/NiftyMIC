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

import matplotlib.pyplot as plt

## Import modules from src-folder
import ReconstructionManager as rm
import SimpleITKHelper as sitkh
# import InPlaneRigidRegistration as inplaneRR


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

    def test_01_transformation_back_to_origin(self):

        ## Fetch data
        stack_number = 0
        slice_number = 0
        stacks = reconstruction_manager.get_stacks()
        slices = stacks[stack_number].get_slices()
        slice_sitk = sitk.Image(slices[slice_number].sitk)

        ## Define vectors pointing along the main axis of the image space
        N_x, N_y, N_z = slice_sitk.GetSize()
        spacing = np.array(slice_sitk.GetSpacing())

        e_0 = (0,0,0)
        e_x = (N_x,0,0)
        e_y = (0,N_y,0)
        e_z = (0,0,N_z)
        e_xyz = (N_x,N_y,N_z)

        ## Extract affine transformation to transform from Image to Physical Space
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_sitk)

        ## T = T_rotation_inv o T_origin_inv
        T = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_sitk)

        ## Test that image axis in physical space now align with physical coordinate system ones
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_0 - T.TransformPoint(slice_sitk.TransformIndexToPhysicalPoint(e_0) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_x - T.TransformPoint(slice_sitk.TransformIndexToPhysicalPoint(e_x) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_y - T.TransformPoint(slice_sitk.TransformIndexToPhysicalPoint(e_y) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_z - T.TransformPoint(slice_sitk.TransformIndexToPhysicalPoint(e_z) ))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_xyz - T.TransformPoint(slice_sitk.TransformIndexToPhysicalPoint(e_xyz) ))
            , decimals = accuracy), 0 )


        ## Transformation representing the one needed to incorporated in the image header
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T, T_PI)

        ## Extract origin and direction to write into axis aligned stack
        origin_PI_align = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align,slice_sitk)
        direction_PI_align = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align,slice_sitk)

        ## Update image header
        slice_sitk.SetDirection(direction_PI_align)
        slice_sitk.SetOrigin(origin_PI_align)

        ## Test whether the update of the image header (set origin and direction) lead to the same results
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_0 - slice_sitk.TransformIndexToPhysicalPoint(e_0) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_x - slice_sitk.TransformIndexToPhysicalPoint(e_x) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_y - slice_sitk.TransformIndexToPhysicalPoint(e_y) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_z - slice_sitk.TransformIndexToPhysicalPoint(e_z) )
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            np.linalg.norm( spacing*e_xyz - slice_sitk.TransformIndexToPhysicalPoint(e_xyz) )
            , decimals = accuracy), 0 )


    def test_02_alignment_with_PI_align_trafo_coordinate_system(self):
        ## Set test data:
        slice_number = 0
        angle = np.pi/3
        translation = (20,30)
        center = (10,15)

        ## Fetch data
        stack_number = 0
        slice_number = 0
        stacks = reconstruction_manager.get_stacks()
        slices = stacks[stack_number].get_slices()
        slice_sitk = sitk.Image(slices[slice_number].sitk)

        ## Define vectors pointing along the main axis of the image space
        N_x, N_y, N_z = slice_sitk.GetSize()
        
        e_0 = (0,0,0)
        e_x = (N_x,0,0)
        e_y = (0,N_y,0)
        e_xy = (N_x,N_y,0)


        ## Extract affine transformation to transform from Image to Physical Space
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_sitk)

        ## T = T_rotation_inv o T_origin_inv
        T = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_sitk)

        ## T_PI_align = T_rotation_inv o T_origin_inv o T_PI: Trafo to align stack with physical coordinate system
        ## (Hence, T_PI_align(i) = spacing*i)
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T, T_PI)

        ## Extract direction matrix and origin so that slice is oriented according to T_PI_align (i.e. with physical axes)
        origin_PI_align = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align,slice_sitk)
        direction_PI_align = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align,slice_sitk)

        ## Update image header
        """
        After this line: direction = eye(3) and origin = \0. Only spacing kept!
        Hence: Find proper trafo by hand do get this without numerical approximation or set it by hand!
        """
        slice_sitk.SetDirection(direction_PI_align)
        slice_sitk.SetOrigin(origin_PI_align)

        ## Fetch slices
        slice_3D = slice_sitk[:,:,slice_number:slice_number+1]        
        slice_2D = slice_sitk[:,:,slice_number]        

        ## Create 2D rigid transform
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Extend to 3D rigid transform
        rigid_transform_3D = sitkh.get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D)  

        ## Check results
        point = e_0
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )

        point = e_x
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )

        point = e_y
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )
        
        point = e_xy
        res_2D = rigid_transform_2D.TransformPoint(slice_2D.TransformIndexToPhysicalPoint((point[0],point[1])))
        self.assertEqual(np.around(
            np.linalg.norm(
                np.array([res_2D[0],res_2D[1],0]) -
                np.array(rigid_transform_3D.TransformPoint(slice_3D.TransformIndexToPhysicalPoint(point)))
                )
            , decimals = accuracy), 0 )


    def test_03_in_plane_rigid_transformation_in_3D_space(self):
        ## Set test data:
        slice_number = 0
        angle = np.pi/3
        translation = (20,30)
        center = (10,0)

        ## Fetch data
        stack_number = 0
        slice_number = 0
        stacks = reconstruction_manager.get_stacks()
        slices = stacks[stack_number].get_slices()
        slice_sitk = sitk.Image(slices[slice_number].sitk)

        ## Create 2D rigid transform
        rigid_transform_2D = sitk.Euler2DTransform(center, angle, translation)

        ## Create deep copy for before/after comparison
        slice_sitk_original = sitk.Image(slice_sitk)

        ## Define vectors pointing along the main axis of the image space
        spacing = np.array(slice_sitk.GetSpacing())
        N_x, N_y, N_z = slice_sitk.GetSize()
        
        e_0 = (0,0,0)
        e_x = (N_x,0,0)
        e_y = (0,N_y,0)
        e_xy = (N_x,N_y,0)
        e_c = (center[0],center[1],0)/spacing

        ## Transform to physical origin
        T = sitkh.get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_sitk)

        ## Get in-plane rigid transform in 3D space
        T_PI_in_plane_rotation_3D = sitkh.get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D, T, slice_sitk)      

        ## Fetch corresponding information for origin and direction
        origin_PI_in_plane_rotation_3D = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_in_plane_rotation_3D, slice_sitk)
        direction_PI_in_plane_rotation_3D = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_in_plane_rotation_3D, slice_sitk)

        ## Update information at slice_sitk
        slice_sitk.SetOrigin(origin_PI_in_plane_rotation_3D)
        slice_sitk.SetDirection(direction_PI_in_plane_rotation_3D)


        ## Test planar alignment of 3D images
        a_0 = np.array(slice_sitk_original.TransformIndexToPhysicalPoint(e_0))
        a_x = np.array(slice_sitk_original.TransformIndexToPhysicalPoint(e_x)) - a_0
        a_y = np.array(slice_sitk_original.TransformIndexToPhysicalPoint(e_y)) - a_0
        a_c = np.array(slice_sitk_original.TransformContinuousIndexToPhysicalPoint(e_c))
        a_z = np.cross(a_x,a_y)

        b_0 = np.array(slice_sitk.TransformIndexToPhysicalPoint(e_0))
        b_x = np.array(slice_sitk.TransformIndexToPhysicalPoint(e_x)) - b_0
        b_y = np.array(slice_sitk.TransformIndexToPhysicalPoint(e_y)) - b_0
        b_c = np.array(slice_sitk.TransformContinuousIndexToPhysicalPoint(e_c)) 

        t_3D = b_c - a_c

        ## Check: Rigid transformation
        ## Not exactly orthogonal!!
        # self.assertEqual(np.around(
        #     abs(a_x.dot(a_y))
        #     , decimals = accuracy), 0 )
        # self.assertEqual(np.around(
        #     abs(b_x.dot(b_y))
        #     , decimals = accuracy), 0 )

        ## Check: a_x-a_y-plane parallel to b_x-b_y-plane
        self.assertEqual(np.around(
            abs(a_z.dot(b_x))
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            abs(a_z.dot(b_y))
            , decimals = accuracy), 0 )

        ## Check: In-plane translation vector (i.e. translation in a_x-b_x-plane)
        self.assertEqual(np.around(
            abs(a_z.dot(t_3D))
            , decimals = accuracy), 0 )

        ## Check: Isometric transformation:
        self.assertEqual(np.around(
            abs(np.linalg.norm(a_x) - np.linalg.norm(b_x))
            , decimals = accuracy-1), 0 )
        self.assertEqual(np.around(
            abs(np.linalg.norm(a_y) - np.linalg.norm(b_y))
            , decimals = accuracy-1), 0 )

        ## Check: Translation vectors have the same norm
        # print("t_2D = " + str(t_2D))
        # print("t_3D = " + str(t_3D))
        # print("||t_2D|| = " + str(np.linalg.norm(t_2D)))
        # print("||t_3D|| = " + str(np.linalg.norm(t_3D)))
        # print("||t_2D|| - ||t_3D|| = " + str(
        #     abs(np.linalg.norm(t_2D) - np.linalg.norm(t_3D))
        #     ))
        t_2D = np.array([translation[0], translation[1]])
        self.assertEqual(np.around(
            abs(np.linalg.norm(t_2D) - np.linalg.norm(t_3D))
            , decimals = accuracy), 0 )

        ## Check: Rotation angle correct:
        alpha_3D_x = np.arccos(a_x.dot(b_x)/(np.linalg.norm(a_x)*np.linalg.norm(b_x)))
        alpha_3D_y = np.arccos(a_y.dot(b_y)/(np.linalg.norm(a_y)*np.linalg.norm(b_y)))

        self.assertEqual(np.around(
            abs(alpha_3D_x - angle)
            , decimals = accuracy), 0 )
        self.assertEqual(np.around(
            abs(alpha_3D_y - angle)
            , decimals = accuracy), 0 )


    def test_04_compare_in_plane_registration_of_single_3D_slices_with_2D_ones(self):
        ## Run in-plane rigid registration

        reconstruction_manager.run_in_plane_rigid_registration()
        in_plane_rigid_registration = reconstruction_manager._in_plane_rigid_registration

        stacks_aligned_3D_sitk = in_plane_rigid_registration._UT_get_resampled_stacks()
        stacks_aligned_2D_sitk = in_plane_rigid_registration._UT_2D_resampled_stacks

        plot = False
        # plot = True

        for i in range(0, len(stacks_aligned_2D_sitk)):
            stacks_aligned_3D_nda = sitk.GetArrayFromImage(stacks_aligned_3D_sitk[i])
            stacks_aligned_2D_nda = sitk.GetArrayFromImage(stacks_aligned_2D_sitk[i])

            N_slices = stacks_aligned_2D_sitk[i].GetDepth()

            if plot:
                for j in range(0, 3):
                # for j in range(0, N_slices):

                    fig = plt.figure(1)
                    plt.suptitle("Slice %r/%r: error (norm) = %r" %(j+1,N_slices,np.linalg.norm(stacks_aligned_3D_nda[j,:,:]-stacks_aligned_2D_nda[j,:,:])))
                    plt.subplot(1,3,1)
                    plt.imshow(stacks_aligned_2D_nda[j,:,:], cmap="Greys_r")
                    plt.axis('off')

                    plt.subplot(1,3,2)
                    plt.imshow(stacks_aligned_3D_nda[j,:,:], cmap="Greys_r")
                    plt.axis('off')

                    plt.subplot(1,3,3)
                    plt.imshow(stacks_aligned_3D_nda[j,:,:]-stacks_aligned_2D_nda[j,:,:], cmap="Greys_r")
                    plt.axis('off')
                    plt.show()

            self.assertEqual(np.around(
                np.linalg.norm(stacks_aligned_3D_nda - stacks_aligned_2D_nda)
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
    dir_output = "results/"

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

    #!
    # dir_input = "../src/GettingStarted/data/"
    # filenames =  ["kidney_s"]
    # filenames =  ["fetal_brain_a"]
    # filenames =  ["fetal_brain_c"]
    # filenames =  ["fetal_brain_s"]
    #!

    reconstruction_manager.read_input_data(dir_input, filenames)

    print("\nUnit tests:\n--------------")
    unittest.main()

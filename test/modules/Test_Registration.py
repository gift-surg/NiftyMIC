## \file Test_Registration.py
#  \brief  Class containing unit tests for module Stack
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2016


## Import libraries 
import SimpleITK as sitk
import itk
import numpy as np
import unittest
import sys

## Add directories to import modules
dir_src_root = "../src/"
sys.path.append( dir_src_root + "base/" )
sys.path.append( dir_src_root + "registration/" )

## Import modules
import Stack as st
import Registration as myreg

## Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

## ITK image type
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]

## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Test_Registration(unittest.TestCase):

    ## Specify input data
    dir_input = "data/registration/"

    accuracy = 4

    def setUp(self):
        pass

    def test_reshaping_of_structures(self):

        # filename_prefix = "RigidTransform_"
        filename_prefix = "TranslationOnly_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"
        stack_sim = st.Stack.from_filename(self.dir_input, filename_StackSim)
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HRVolume)

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        itk2np = itk.PyBuffer[itk.Image.D3]
        itk2np_CVD33 = itk.PyBuffer[itk.Image.CVD33]

        filter_OrientedGaussian_3D = itk.OrientedGaussianInterpolateImageFilter[IMAGE_TYPE, IMAGE_TYPE].New()
        filter_OrientedGaussian_3D.SetInput(HR_volume.itk)
        filter_OrientedGaussian_3D.SetUseJacobian(True)
        
        for j in range(0, N_slices):         
            slice = slices_sim[j]

            filter_OrientedGaussian_3D.SetOutputParametersFromImage(slice.itk)
            filter_OrientedGaussian_3D.UpdateLargestPossibleRegion()
            filter_OrientedGaussian_3D.Update()

            slice_simulated_nda  = itk2np.GetArrayFromImage(filter_OrientedGaussian_3D.GetOutput())
            dslice_simulated_nda = itk2np_CVD33.GetArrayFromImage(filter_OrientedGaussian_3D.GetJacobian())

            shape = slice_simulated_nda.shape

            slice_simulated_nda_flat = slice_simulated_nda.flatten()
            dslice_simulated_nda_flat = dslice_simulated_nda.reshape(-1,3)

            array0 = np.zeros(3)
            array1 = np.zeros(3)
            abs_diff = 0

            iter = 0
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    for k in range(0, shape[2]):
                        array0 = slice_simulated_nda[i,j,k] - dslice_simulated_nda[i,j,k,:]
                        array1 = slice_simulated_nda_flat[iter] - dslice_simulated_nda_flat[iter,:]
                        abs_diff += np.linalg.norm(array0-array1) 
                        iter +=1 

            self.assertEqual(np.round(
                abs_diff
            , decimals = self.accuracy), 0)
            
        

    def test_registration_of_slices(self):

        filename_prefix = "RigidTransform_"
        # filename_prefix = "TranslationOnly_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"
        stack_sim = st.Stack.from_filename(self.dir_input, filename_StackSim)
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HRVolume)

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        for j in range(0, N_slices):
            rigid_transform_groundtruth_sitk = sitk.ReadTransform(self.dir_input + filename_transforms_prefix + str(j) + ".tfm")
            parameters_gd = np.array(rigid_transform_groundtruth_sitk.GetParameters())

            registration = myreg.Registration(slices_sim[j], HR_volume)
            registration.run_registration()
            # registration.print_statistics()
            parameters = registration.get_parameters()

            norm_diff = np.linalg.norm(parameters-parameters_gd)
            print("Slice %s/%s: |parameters-parameters_gd| = %s" %(j, N_slices-1, str(norm_diff)) )

            self.assertEqual(np.round(
                norm_diff
                , decimals = self.accuracy), 0)
        

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
import time
from datetime import timedelta

## Add directories to import modules
dir_src_root = "../src/"
sys.path.append( dir_src_root + "base/" )
sys.path.append( dir_src_root + "registration/" )

## Import modules
import Stack as st
import Registration as myreg
import SimpleITKHelper as sitkh
import RegistrationSimpleITK as regsitk
import RegistrationITK as regitk

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


    def test_translation_registration_of_slices(self):

        filename_prefix = "TranslationOnly_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"
        stack_sim = st.Stack.from_filename(self.dir_input, filename_StackSim)
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HRVolume)

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        time_start = time.time()

        for j in range(0, N_slices):
            rigid_transform_groundtruth_sitk = sitk.ReadTransform(self.dir_input + filename_transforms_prefix + str(j) + ".tfm")
            parameters_gd = np.array(rigid_transform_groundtruth_sitk.GetParameters())

            registration = myreg.Registration(fixed=slices_sim[j], moving=HR_volume)
            registration.run_registration()
            # registration.print_statistics()

            ## Check parameters
            parameters = registration.get_parameters()

            norm_diff = np.linalg.norm(parameters-parameters_gd)
            # print("Slice %s/%s: |parameters-parameters_gd| = %s" %(j, N_slices-1, str(norm_diff)) )

            self.assertEqual(np.round(
                norm_diff
                , decimals = self.accuracy), 0)
        
        ## Set elapsed time
        time_end = time.time()
        # print("Translation only registration test: Elapsed time = %s" %(timedelta(seconds=time_end-time_start)))


    def test_rigid_registration_of_slices(self):

        filename_prefix = "RigidTransform_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"
        stack_sim = st.Stack.from_filename(self.dir_input, filename_StackSim)
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HRVolume)

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        time_start = time.time()

        for j in range(0, N_slices):
            rigid_transform_groundtruth_sitk = sitk.ReadTransform(self.dir_input + filename_transforms_prefix + str(j) + ".tfm")
            parameters_gd = np.array(rigid_transform_groundtruth_sitk.GetParameters())

            registration = myreg.Registration(fixed=slices_sim[j], moving=HR_volume)
            registration.run_registration()
            # registration.print_statistics()

            ## Check parameters
            parameters = registration.get_parameters()

            norm_diff = np.linalg.norm(parameters-parameters_gd)
            # print("Slice %s/%s: |parameters-parameters_gd| = %s" %(j, N_slices-1, str(norm_diff)) )

            self.assertEqual(np.round(
                norm_diff
                , decimals = self.accuracy), 0)

        ## Set elapsed time
        time_end = time.time()
        # print("Rigid registration test: Elapsed time = %s" %(timedelta(seconds=time_end-time_start)))


    def test_rigid_registration_of_stack(self):
        filename_prefix = "NoMotion_"
        parameters_gd = (0.1,0.1,0.2,-1,3,2)

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        stack_sim = st.Stack.from_filename(self.dir_input, filename_StackSim)
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HRVolume)

        ## Apply motion
        transform_sitk = sitk.Euler3DTransform()
        transform_sitk.SetParameters(parameters_gd)
        stack_sitk = sitkh.get_transformed_image(stack_sim.sitk, transform_sitk)
        stack_sitk_mask = sitkh.get_transformed_image(stack_sim.sitk_mask, transform_sitk)
        stack_sim = st.Stack.from_sitk_image(stack_sitk, name=stack_sim.get_filename(), image_sitk_mask=stack_sitk_mask)

        time_start = time.time()
        registration = myreg.Registration(fixed=stack_sim, moving=HR_volume)
        registration.use_verbose(False)
        registration.run_registration()

        ## Check parameters (should be the negative of parameters_gd)
        parameters = registration.get_parameters()
        norm_diff = np.linalg.norm(parameters+parameters_gd)
        print("parameters = " + str(parameters))
        print("|parameters-parameters_gd| = %s" %(str(norm_diff)) )

        self.assertEqual(np.round(
            norm_diff*0.1
            , decimals = 0), 0)

        ## Set elapsed time
        time_end = time.time()
        print("Rigid registration test: Elapsed time = %s" %(timedelta(seconds=time_end-time_start)))


        ######### Comparison with ITK registrations
        scales_estimator = "PhysicalShift"
        use_verbose = False

        ## SimpleITK registration for comparison
        print("SimpleITK registration for comparison:")
        time_start = time.time()
        registration = regsitk.RegistrationSimpleITK(fixed=stack_sim, moving=HR_volume)
        registration.use_verbose(use_verbose)
        registration.set_scales_estimator(scales_estimator)
        registration.run_registration()
        time_end = time.time()

        parameters = np.array(registration.get_registration_transform_sitk().GetParameters())
        norm_diff = np.linalg.norm(parameters+parameters_gd)
        print("\tparameters = " + str(parameters))
        print("\t|parameters-parameters_gd| = %s" %(str(norm_diff)) )
        print("\tRigid registration test: Elapsed time = %s" %(timedelta(seconds=time_end-time_start)))

        ## ITK registration for comparison
        print("ITK registration for comparison:")
        time_start = time.time()
        registration = regitk.RegistrationITK(fixed=stack_sim, moving=HR_volume)
        registration.use_verbose(use_verbose)
        registration.set_scales_estimator(scales_estimator)
        registration.run_registration()
        time_end = time.time()

        parameters = np.array(registration.get_registration_transform_sitk().GetParameters())
        norm_diff = np.linalg.norm(parameters+parameters_gd)
        print("\tparameters = " + str(parameters))
        print("\t|parameters-parameters_gd| = %s" %(str(norm_diff)) )
        print("\tRigid registration test: Elapsed time = %s" %(timedelta(seconds=time_end-time_start)))

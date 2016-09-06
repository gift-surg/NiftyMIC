## \file Test_Registration.py
#  \brief  Class containing unit tests for module Stack
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2016


## Import libraries 
import SimpleITK as sitk
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

## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Test_Registration(unittest.TestCase):

    ## Specify input data
    dir_input = "data/registration/"

    accuracy = 5

    def setUp(self):
        pass

    def test_registration_of_slices(self):

        filename_HRVolume = "HRVolume"
        filename_StackSim = "StackSimulated"
        filename_transforms_prefix = "RigidTransformGroundTruth_slice"
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
        

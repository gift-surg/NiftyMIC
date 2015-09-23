## Import libraries:
import SimpleITK as sitk
import numpy as np
import nibabel as nib 
import unittest

import sys
sys.path.append("../")

## Import modules from src-folder:
import SimpleITKHelper as sitkh

"""
Functions
"""

"""
Unit Test Class
"""
## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug accuracy, 2015
class TestUM(unittest.TestCase):

    def setUp(self):
        pass

    def test_01_check_2D_extraction_of_3D_origin(self):
        for i in range(0,N_stacks):
            origin_3D = np.array(stack[i][:,:,0:1].GetOrigin())
            origin_2D = np.array(stack[i][:,:,0].GetOrigin())

            # print("\nStack %r" %i)
            # print("origin_2D = %s" % str(origin_2D))
            self.assertEqual(np.around(
                np.sum(abs(origin_3D[0:-1] - origin_2D))
                , decimals = accuracy), 0 )

    def test_02_check_2D_extraction_of_3D_direction(self):
        for i in range(0,N_stacks):
            direction_3D = np.array(stack[i][:,:,0:1].GetDirection()).reshape(3,3)
            direction_2D = np.array(stack[i][:,:,0].GetDirection()).reshape(2,2)

            print("\nStack %r" %i)
            print("det direction_3D = %r " % np.linalg.det(direction_3D))
            print direction_3D
            print("det direction_2D = %r" % np.linalg.det(direction_2D))
            print direction_2D

            self.assertEqual(np.around(
                np.sum(abs(direction_3D[0:-1,0:-1] - direction_2D))
                , decimals = accuracy), 0 )

    

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
    # filenames = ["fetal_brain_a", "fetal_brain_c", "fetal_brain_s"]
    filenames = ["fetal_brain_a", "fetal_brain_c", "fetal_brain_s", "placenta_s"]

    accuracy = 6 # decimal places for accuracy of unit tests

    """
    Fetch data
    """
    N_stacks = len(filenames)
    
    ## Read images 
    stack = [None]*N_stacks
    for i in range(0,N_stacks):
        stack[i] = sitk.ReadImage(dir_input+filenames[i]+".nii.gz", sitk.sitkFloat64)

    """
    Playground
    """
    ## names for working in ipython
    stack_fine = stack[2]
    stack_error = stack[3]

    slice_number = i

    slice_3D_fine = stack_fine[:,:,i:i+1]
    direction_3D_fine = np.array(slice_3D_fine.GetDirection()).reshape(3,3)
    origin_3D_fine = np.array(slice_3D_fine.GetOrigin())

    slice_3D_error = stack_error[:,:,i:i+1]
    direction_3D_error = np.array(slice_3D_error.GetDirection()).reshape(3,3)
    origin_3D_error = np.array(slice_3D_error.GetOrigin())

    slice_2D_fine = stack_fine[:,:,i]
    direction_2D_fine = np.array(slice_2D_fine.GetDirection()).reshape(2,2)
    origin_2D_fine = np.array(slice_2D_fine.GetOrigin())

    slice_2D_error = stack_error[:,:,i]
    direction_2D_error = np.array(slice_2D_error.GetDirection()).reshape(2,2)
    origin_2D_error = np.array(slice_2D_error.GetOrigin())



    """
    Unit tests
    """
    e_0 = np.array((0,0,0))
    e_x = np.array((1,0,0))
    e_y = np.array((0,1,0))
    e_z = np.array((0,0,1))

    print("\nUnit tests:\n--------------")
    unittest.main()
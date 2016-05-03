## \file Test_SimpleITKHelper.py
#  \brief  Class containing unit tests for module SimpleITKHelper
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date December 2015


# Import libraries 
import SimpleITK as sitk
import itk
import numpy as np
import unittest

import sys
sys.path.append("../src/")
sys.path.append("data/SimpleITKHelper/")

## Import modules from src-folder
# import FirstEstimateOfHRVolume as tm
# import StackManager as sm
# import Slice as sl
import SimpleITKHelper as sitkh



## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class Test_SimpleITKHelper(unittest.TestCase):

    ## Specify input data
    dir_input = "data/SimpleITKHelper/"

    accuracy = 8

    def setUp(self):
        pass

    def test_get_correct_itk_orientation_from_sitk_image(self):
        filename = "stack0"
        # filename = "stack0_rotated_angle_z_is_pi_over_10"

        ## Read image via sitk
        image_sitk = sitk.ReadImage(self.dir_input + filename + ".nii.gz")

        ## Read image via itk
        dimension = 3
        pixel_type = itk.D
        image_type = itk.Image[pixel_type, dimension]
        reader_type = itk.ImageFileReader[image_type]
        image_IO = itk.NiftiImageIO.New()

        reader = reader_type.New()
        reader.SetImageIO(image_IO)
        reader.SetFileName(self.dir_input + filename + ".nii.gz")
        reader.Update()
        image_itk = reader.GetOutput()

        ## Change header information of sitk image
        origin = (0,0,0)
        direction = (1,0,0, 0,1,0, 0,0,1)
        spacing = (1,1,1)

        image_sitk.SetSpacing(spacing)
        image_sitk.SetDirection(direction)
        image_sitk.SetOrigin(origin)

        ## Update header of itk image
        image_itk.SetOrigin(image_sitk.GetOrigin())
        image_itk.SetDirection(sitkh.get_itk_direction_from_sitk_image(image_sitk))
        image_itk.SetSpacing(image_sitk.GetSpacing())

        ## Write itk image and read it again as sitk image
        writer = itk.ImageFileWriter[image_type].New()
        writer.SetFileName("/tmp/itk_update.nii.gz")
        writer.SetInput(image_itk)
        writer.Update()

        image_sitk_from_itk = sitk.ReadImage("/tmp/itk_update.nii.gz")

        ## Check origin
        self.assertEqual(np.around(
            np.linalg.norm(np.array(image_sitk_from_itk.GetOrigin()) - image_sitk.GetOrigin())
            , decimals = self.accuracy), 0 )

        ## Check spacing
        self.assertEqual(np.around(
            np.linalg.norm(np.array(image_sitk_from_itk.GetSpacing()) - image_sitk.GetSpacing())
            , decimals = self.accuracy), 0 )

        ## Check direction matrix
        self.assertEqual(np.around(
            np.linalg.norm(np.array(image_sitk_from_itk.GetDirection()) - image_sitk.GetDirection())
            , decimals = self.accuracy), 0 )


    def test_get_sitk_affine_transform_from_sitk_direction_and_origin(self):
        filename = "stack0"

        ## Read image via sitk
        image_sitk = sitk.ReadImage(self.dir_input + filename + ".nii.gz")

        origin = image_sitk.GetOrigin()
        direction = image_sitk.GetDirection()

        affine_transform_ref = sitkh.get_sitk_affine_transform_from_sitk_image(image_sitk)
        affine_transform = sitkh.get_sitk_affine_transform_from_sitk_direction_and_origin(direction, origin, image_sitk)

        ## Check Fixed Parameters
        self.assertEqual(np.around(
            np.linalg.norm(np.array(affine_transform_ref.GetFixedParameters()) - affine_transform.GetFixedParameters())
            , decimals = self.accuracy), 0 )

        ## Check Parameters
        self.assertEqual(np.around(
            np.linalg.norm(np.array(affine_transform_ref.GetParameters()) - affine_transform.GetParameters())
            , decimals = self.accuracy), 0 )



## \file Test_SliceAcqusition.py
#  \brief  Class containing unit tests for module SliceAcqusition
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2016


## Import libraries 
import SimpleITK as sitk
import itk
import numpy as np
import unittest
import sys
sys.path.append("../src/")
sys.path.append("../studies/")
sys.path.append("data/")

## Import modules from src-folder
import SimpleITKHelper as sitkh
import Stack as st
import SliceAcqusition as sa
import PSF as psf


## Pixel type of used 3D ITK image
pixel_type = itk.D

## ITK image type 
image_type = itk.Image[pixel_type, 3]


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class SliceAcqusition(unittest.TestCase):

    ## Specify input data
    dir_input = "data/"

    accuracy = 7

    def setUp(self):
        pass


    def test_conversion_image_direction(self):

        filename_HR_volume = "recon_fetal_neck_mass_brain_cycles0_SRR_TK0_itermax20_alpha0.1"
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HR_volume)

        ## Get unit vectors defining image grid in physical space and construct direction matrix
        origin_HR_volume = np.array(HR_volume.sitk.GetOrigin())
        a_x = HR_volume.sitk.TransformIndexToPhysicalPoint((1,0,0)) - origin_HR_volume
        a_y = HR_volume.sitk.TransformIndexToPhysicalPoint((0,1,0)) - origin_HR_volume
        a_z = HR_volume.sitk.TransformIndexToPhysicalPoint((0,0,1)) - origin_HR_volume

        e_x = a_x/np.linalg.norm(a_x)
        e_y = a_y/np.linalg.norm(a_y)
        e_z = a_z/np.linalg.norm(a_z)

        direction_matrix_test = np.array([e_x, e_y, e_z]).transpose()
        direction_test = direction_matrix_test.flatten()

        ## Get respective vectors from image direction
        direction = np.array(HR_volume.sitk.GetDirection())
        
        e_x_test = direction[0::3]
        e_y_test = direction[1::3]
        e_z_test = direction[2::3]

        ## Check correspondences
        self.assertEqual(np.round( np.linalg.norm(e_x_test - e_x), decimals = self.accuracy), 0)
        self.assertEqual(np.round( np.linalg.norm(e_y_test - e_y), decimals = self.accuracy), 0)
        self.assertEqual(np.round( np.linalg.norm(e_z_test - e_z), decimals = self.accuracy), 0)
        
        self.assertEqual(np.round( np.linalg.norm(direction - direction_test), decimals = self.accuracy), 0)


    def test_run_simulation_view(self):

        filename_HR_volume = "recon_fetal_neck_mass_brain_cycles0_SRR_TK0_itermax20_alpha0.1"
        HR_volume = st.Stack.from_filename(self.dir_input, filename_HR_volume)

        ## 1) Test for Nearest Neighbor Interpolator
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("NearestNeighbor")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        stacks_simulated = slice_acquistion.get_simulated_stacks()

        for i in range(0, len(stacks_simulated)):
            HR_volume_resampled_sitk = sitk.Resample(
                HR_volume.sitk, stacks_simulated[i].sitk, sitk.Euler3DTransform(), sitk.sitkNearestNeighbor, 0.0, stacks_simulated[i].sitk.GetPixelIDValue()
                )

        self.assertEqual(np.round(
            np.linalg.norm(sitk.GetArrayFromImage(stacks_simulated[i].sitk - HR_volume_resampled_sitk))
            , decimals = self.accuracy), 0)


        ## 2) Test for Linear Interpolator
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("Linear")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        stacks_simulated = slice_acquistion.get_simulated_stacks()

        for i in range(0, len(stacks_simulated)):
            HR_volume_resampled_sitk = sitk.Resample(
                HR_volume.sitk, stacks_simulated[i].sitk, sitk.Euler3DTransform(), sitk.sitkLinear, 0.0, stacks_simulated[i].sitk.GetPixelIDValue()
                )

        self.assertEqual(np.round(
            np.linalg.norm(sitk.GetArrayFromImage(stacks_simulated[i].sitk - HR_volume_resampled_sitk))
            , decimals = self.accuracy), 0)


        ## 3) Test for Oriented Gaussian Interpolator
        slice_acquistion = sa.SliceAcqusition(HR_volume)
        slice_acquistion.set_interpolator_type("OrientedGaussian")

        slice_acquistion.run_simulation_view_1()
        slice_acquistion.run_simulation_view_2()
        slice_acquistion.run_simulation_view_3()

        stacks_simulated = slice_acquistion.get_simulated_stacks()

        resampler = itk.ResampleImageFilter[image_type, image_type].New()
        resampler.SetDefaultPixelValue( 0.0 )
        resampler.SetInput( HR_volume.itk )

        interpolator = itk.OrientedGaussianInterpolateImageFunction[image_type, pixel_type].New()
        alpha_cut = 3
        interpolator.SetAlpha(alpha_cut)
        resampler.SetInterpolator(interpolator)

        PSF = psf.PSF()

        for i in range(0, len(stacks_simulated)):
            resampler.SetOutputParametersFromImage( stacks_simulated[i].itk )
            
            ## Set covariance based on oblique PSF
            Cov_HR_coord = PSF.get_gaussian_PSF_covariance_matrix_HR_volume_coordinates(stacks_simulated[i], HR_volume)
            interpolator.SetCovariance(Cov_HR_coord.flatten())

            resampler.UpdateLargestPossibleRegion()
            resampler.Update()

            HR_volume_resampled_itk = resampler.GetOutput()
            HR_volume_resampled_sitk = sitkh.convert_itk_to_sitk_image(HR_volume_resampled_itk)

        self.assertEqual(np.round(
            np.linalg.norm(sitk.GetArrayFromImage(stacks_simulated[i].sitk - HR_volume_resampled_sitk))
            , decimals = self.accuracy), 0)
        

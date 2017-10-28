##
# \file installation_test.py
#  \brief  Class to test installation
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date October 2017


# Import libraries
import unittest

from nipype.testing import example_data

import niftymic.base.stack as st
import niftymic.registration.cpp_itk_registration as cppreg
import niftymic.registration.flirt as flirt
import niftymic.registration.niftyreg as niftyreg
import niftymic.utilities.brain_stripping as bs


class InstallationTest(unittest.TestCase):

    def setUp(self):

        self.accuracy = 10

        self.path_to_fixed = example_data("segmentation0.nii.gz")
        self.path_to_moving = example_data("segmentation1.nii.gz")

        self.fixed = st.Stack.from_filename(self.path_to_fixed)
        self.moving = st.Stack.from_filename(self.path_to_moving)

    ##
    # Test whether FSL installation was successful
    # \date       2017-10-26 15:02:44+0100
    #
    def test_fsl(self):

        # Run flirt registration
        registration_method = flirt.FLIRT(
            fixed=self.fixed, moving=self.moving)
        registration_method.run()

        # Run BET brain stripping
        brain_stripper = bs.BrainStripping.from_sitk_image(self.fixed.sitk)
        brain_stripper.run()

    ##
    # Test whether NiftyReg installation was successful
    # \date       2017-10-26 15:08:59+0100
    #
    def test_niftyreg(self):

        # Run reg_aladin registration
        registration_method = niftyreg.RegAladin(
            fixed=self.fixed, moving=self.moving)
        registration_method.run()

        # Run reg_f3d registration
        registration_method = niftyreg.RegF3D(
            fixed=self.fixed, moving=self.moving)
        registration_method.run()

    ##
    # Test whether ITK_NiftyMIC installation was successful
    # \date       2017-10-26 15:12:26+0100
    #
    def test_itk_niftymic(self):

        import itk
        image_itk = itk.Image.D3.New()
        filter_itk = itk.OrientedGaussianInterpolateImageFilter.ID3ID3.New()

    ##
    # Test whether command line interfaces have been installed successfully
    # \date       2017-10-26 15:26:34+0100
    #
    def test_command_line_interfaces(self):

        registration_method = cppreg.CppItkRegistration(
            fixed=self.fixed, moving=self.moving)
        registration_method.run()

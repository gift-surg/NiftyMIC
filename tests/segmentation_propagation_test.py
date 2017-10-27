# \file TestSegmentationPropagation.py
#  \brief  Class containing unit tests for module SegmentationPropagation
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date May 2017


import unittest

# Import libraries
import SimpleITK as sitk
import numpy as np
import os

import pysitk.simple_itk_helper as sitkh
import niftymic.base.stack as st
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.utilities.segmentation_propagation as segprop
from niftymic.definitions import DIR_TEST


class SegmentationPropagationTest(unittest.TestCase):

    # Specify input data
    dir_test_data = DIR_TEST

    accuracy = 6

    def setUp(self):
        pass

    def test_registration(self):

        filename = "fetal_brain_0"

        parameters_gd = (0.1, 0.2, -0.3, 0, -4, 10)

        template = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename + ".nii.gz"),
            os.path.join(self.dir_test_data, filename + "_mask.nii.gz"),
            extract_slices=False)

        transform_sitk_gd = sitk.Euler3DTransform()
        transform_sitk_gd.SetParameters(parameters_gd)

        stack_sitk = sitkh.get_transformed_sitk_image(
            template.sitk, transform_sitk_gd)

        stack = st.Stack.from_sitk_image(stack_sitk,
                                         filename="stack")

        # sitkh.show_sitk_image([template.sitk, stack_sitk])

        optimizer = "RegularStepGradientDescent"
        optimizer_params = {
            'learningRate': 1,
            'minStep': 1e-6,
            'numberOfIterations': 300
        }

        # optimizer="ConjugateGradientLineSearch"
        # optimizer_params="{'learningRate': 1, 'numberOfIterations': 100}"

        registration = regsitk.SimpleItkRegistration(
            initializer_type="MOMENTS",
            use_verbose=True,
            metric="MeanSquares",
            optimizer=optimizer,
            optimizer_params=optimizer_params,
            use_multiresolution_framework=True,
        )
        # registration = regitk.CppItkRegistration()
        # registration = regniftyreg.RegAladin()

        segmentation_propagation = segprop.SegmentationPropagation(
            stack=stack,
            template=template,
            registration_method=registration,
            dilation_radius=10,
        )
        segmentation_propagation.run_segmentation_propagation()
        foo = segmentation_propagation.get_segmented_stack()
        # sitkh.show_stacks(
        #     [template, foo, stack],
        #     label=["template", "stack_prop", "stack_orig"]
        # )

        transform = segmentation_propagation.get_registration_transform_sitk()
        parameters = sitk.Euler3DTransform(
            transform.GetInverse()).GetParameters()

        self.assertEqual(np.round(
            np.linalg.norm(np.array(parameters) - parameters_gd), decimals=0), 0)

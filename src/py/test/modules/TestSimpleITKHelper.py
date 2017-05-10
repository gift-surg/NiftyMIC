## \file TestSimpleITKHelper.py
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

## Import modules
import utilities.SimpleITKHelper as sitkh
import base.Stack as st

from definitions import dir_test

def get_affine_registration_transform(fixed_sitk, moving_sitk, fixed_mask_sitk=None, moving_mask_sitk=None, print_info=False):

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    # initial_transform = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    # initial_transform = sitk.Euler3DTransform()
    initial_transform = sitk.AffineTransform(3)

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set interpolator to use
    registration_method.SetInterpolator(sitk.sitkLinear)

    ## Set an image masks in order to restrict the sampled points for the metric
    if fixed_mask_sitk is not None:
        registration_method.SetMetricFixedMask(fixed_mask_sitk)
    if moving_mask_sitk is not None:
        registration_method.SetMetricMovingMask(moving_mask_sitk)

    """
    similarity metric settings
    """
    ## Use normalized cross correlation using a small neighborhood for each voxel between two images, with speed optimizations for dense registration
    # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5)
    
    ## Use negative normalized cross correlation image metric
    # registration_method.SetMetricAsCorrelation()

    ## Use demons image metric
    # registration_method.SetMetricAsDemons(intensityDifferenceThreshold=1e-3)

    ## Use mutual information between two images
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50, varianceForJointPDFSmoothing=3)
    
    ## Use the mutual information between two images to be registered using the method of Mattes2001
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    ## Use negative means squares image metric
    registration_method.SetMetricAsMeanSquares()
    
    """
    optimizer settings
    """
    ## Set optimizer to Nelder-Mead downhill simplex algorithm
    # registration_method.SetOptimizerAsAmoeba(simplexDelta=0.1, numberOfIterations=100, parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4, withRestarts=False)

    ## Conjugate gradient descent optimizer with a golden section line search for nonlinear optimization
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

    ## Set the optimizer to sample the metric at regular steps
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=50, stepLength=1.0)

    ## Gradient descent optimizer with a golden section line search
    # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    ## Limited memory Broyden Fletcher Goldfarb Shannon minimization with simple bounds
    # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, maximumNumberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e+7)

    ## Regular Step Gradient descent optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.01, numberOfIterations=100)

    ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
    ## (Many more possibilities to estimate scales)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    """
    setup for the multi-resolution framework            
    """
    ## Set the shrink factors for each level where each level has the same shrink factor for each dimension
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

    ## Enable the smoothing sigmas for each level in physical units (default) or in terms of voxels (then *UnitsOff instead)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    ## Connect all of the observers so that we can perform plotting during registration
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    # print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    # print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    # print("\n")

    ## Execute 3D registration
    final_transform_3D_sitk = registration_method.Execute(fixed_sitk, moving_sitk) 
    if print_info:
        print("SimpleITK Image Registration Method (Affine):")
        print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return sitk.AffineTransform(final_transform_3D_sitk)


def get_rigid_registration_transform(fixed_sitk, moving_sitk, fixed_mask_sitk=None, moving_mask_sitk=None, print_info=False):

    ## Instantiate interface method to the modular ITKv4 registration framework
    registration_method = sitk.ImageRegistrationMethod()

    ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
    # initial_transform = sitk.CenteredTransformInitializer(fixed_sitk, moving_sitk, sitk.Euler3DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
    initial_transform = sitk.Euler3DTransform()
    # initial_transform = sitk.AffineTransform(3)

    ## Set the initial transform and parameters to optimize
    registration_method.SetInitialTransform(initial_transform)

    ## Set interpolator to use
    registration_method.SetInterpolator(sitk.sitkLinear)

    ## Set an image masks in order to restrict the sampled points for the metric
    if fixed_mask_sitk is not None:
        registration_method.SetMetricFixedMask(fixed_mask_sitk)
    if moving_mask_sitk is not None:
        registration_method.SetMetricMovingMask(moving_mask_sitk)

    """
    similarity metric settings
    """
    ## Use normalized cross correlation using a small neighborhood for each voxel between two images, with speed optimizations for dense registration
    # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5)
    
    ## Use negative normalized cross correlation image metric
    # registration_method.SetMetricAsCorrelation()

    ## Use demons image metric
    # registration_method.SetMetricAsDemons(intensityDifferenceThreshold=1e-3)

    ## Use mutual information between two images
    # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50, varianceForJointPDFSmoothing=3)
    
    ## Use the mutual information between two images to be registered using the method of Mattes2001
    # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

    ## Use negative means squares image metric
    registration_method.SetMetricAsMeanSquares()
    
    """
    optimizer settings
    """
    ## Set optimizer to Nelder-Mead downhill simplex algorithm
    # registration_method.SetOptimizerAsAmoeba(simplexDelta=0.1, numberOfIterations=100, parametersConvergenceTolerance=1e-8, functionConvergenceTolerance=1e-4, withRestarts=False)

    ## Conjugate gradient descent optimizer with a golden section line search for nonlinear optimization
    # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)

    ## Set the optimizer to sample the metric at regular steps
    # registration_method.SetOptimizerAsExhaustive(numberOfSteps=50, stepLength=1.0)

    ## Gradient descent optimizer with a golden section line search
    # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)

    ## Limited memory Broyden Fletcher Goldfarb Shannon minimization with simple bounds
    # registration_method.SetOptimizerAsLBFGSB(gradientConvergenceTolerance=1e-5, maximumNumberOfIterations=500, maximumNumberOfCorrections=5, maximumNumberOfFunctionEvaluations=200, costFunctionConvergenceFactor=1e+7)

    ## Regular Step Gradient descent optimizer
    registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=0.01, numberOfIterations=100)

    ## Estimating scales of transform parameters a step sizes, from the maximum voxel shift in physical space caused by a parameter change
    ## (Many more possibilities to estimate scales)
    registration_method.SetOptimizerScalesFromPhysicalShift()
    
    """
    setup for the multi-resolution framework            
    """
    ## Set the shrink factors for each level where each level has the same shrink factor for each dimension
    registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])

    ## Set the sigmas of Gaussian used for smoothing at each level
    registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])

    ## Enable the smoothing sigmas for each level in physical units (default) or in terms of voxels (then *UnitsOff instead)
    registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

    ## Connect all of the observers so that we can perform plotting during registration
    # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
    # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
    # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
    # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

    # print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
    # print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
    # print("\n")

    ## Execute 3D registration
    final_transform_3D_sitk = registration_method.Execute(fixed_sitk, moving_sitk) 
    if print_info:
        print("SimpleITK Image Registration Method (Rigid):")
        print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
        print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))

    return sitk.Euler3DTransform(final_transform_3D_sitk)


## Concept of unit testing for python used in here is based on
#  http://pythontesting.net/framework/unittest/unittest-introduction/
#  Retrieved: Aug 6, 2015
class TestSimpleITKHelper(unittest.TestCase):

    ## Specify input data
    dir_test_data =  dir_test

    accuracy = 6

    def setUp(self):
        pass

    def test_get_correct_itk_orientation_from_sitk_image(self):
        filename = "stack1"
        # filename = "stack1_rotated_angle_z_is_pi_over_10"

        ## Read image via sitk
        image_sitk = sitk.ReadImage(self.dir_test_data + filename + ".nii.gz")

        ## Read image via itk
        dimension = 3
        pixel_type = itk.D
        image_type = itk.Image[pixel_type, dimension]
        reader_type = itk.ImageFileReader[image_type]
        image_IO = itk.NiftiImageIO.New()

        reader = reader_type.New()
        reader.SetImageIO(image_IO)
        reader.SetFileName(self.dir_test_data + filename + ".nii.gz")
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


    ## Test conversion from sitk direction and sitk origin to sitk affine transform
    def test_get_sitk_affine_transform_from_sitk_direction_and_origin(self):
        filename = "stack1"

        ## Read image via sitk
        image_sitk = sitk.ReadImage(self.dir_test_data + filename + ".nii.gz")

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


    def test_get_sitk_affine_transform_from_sitk_image(self):
        filename = "stack1"

        ## Read image via sitk
        image_sitk = sitk.ReadImage(self.dir_test_data + filename + ".nii.gz")

        ## Get indices with intensities greater than 0
        #  indices \in \R^{dim, N_points}
        indices = np.array(np.where(sitk.GetArrayFromImage(image_sitk)[::-1]>0))
        N_points = indices.shape[1]

        ## Get sitk affine transform from image first (the way it is used in Slice.py)
        affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(image_sitk)

        for i in range(0, N_points):
            index = indices[:,i]

            ## Check Alignment
            self.assertEqual(np.around(
                np.linalg.norm(np.array(affine_transform_sitk.TransformPoint(index)) - image_sitk.TransformIndexToPhysicalPoint(index))
                , decimals = self.accuracy), 0 )


    ## Test whether physical position of voxel obtained via affine transform
    #  corresponds to the one obtained by the image directly
    def test_computation_point_physical_space_via_affine_transform(self):
        filename = "stack1"

        ## Read image via sitk
        image_sitk = sitk.ReadImage(self.dir_test_data + filename + ".nii.gz")

        ## Get indices with intensities greater than 0
        #  indices \in \R^{dim, N_points}
        indices = np.array(np.where(sitk.GetArrayFromImage(image_sitk)[::-1]>0))
        N_points = indices.shape[1]

        ## Get sitk affine transform from image first (the way it is used in Slice.py)
        affine_transform_sitk = sitkh.get_sitk_affine_transform_from_sitk_image(image_sitk)

        dim = image_sitk.GetDimension()
        A = np.array(affine_transform_sitk.GetMatrix()).reshape(dim,dim)
        t = np.array(affine_transform_sitk.GetTranslation()).reshape(3,1)

        for i in range(0, N_points):
            index = indices[:,i].reshape(dim,1)

            ## Check Alignment
            self.assertEqual(np.around(
                np.linalg.norm((A.dot(index) + t).flatten() - image_sitk.TransformIndexToPhysicalPoint(index.flatten()))
                , decimals = self.accuracy), 0 )


    ## Test whether \p get_transformed_sitk_image works correct in the rigid and
    #  more general affine case
    def test_get_transformed_sitk_image(self):
        moving_str = "placenta_0"
        fixed_str = "placenta_1"

        moving = st.Stack.from_filename(self.dir_test_data, moving_str, "_mask")
        fixed = st.Stack.from_filename(self.dir_test_data, fixed_str, "_mask")

        ## Register images
        # affine_transform_sitk = get_affine_registration_transform(fixed.sitk, moving.sitk, fixed.sitk_mask, moving.sitk_mask)
        # rigid_transform_sitk = get_rigid_registration_transform(fixed.sitk, moving.sitk, fixed.sitk_mask, moving.sitk_mask)
        affine_transform_sitk = get_affine_registration_transform(fixed.sitk, moving.sitk)
        rigid_transform_sitk = get_rigid_registration_transform(fixed.sitk, moving.sitk)

        interpolator = sitk.sitkBSpline
        default_pixel_value = 0.0

        ## 1) Rigid case
        ## Resample based on obtained registration trafo
        moving_rigidly_warped_sitk = sitk.Resample(moving.sitk, fixed.sitk, rigid_transform_sitk, interpolator, default_pixel_value, moving.sitk.GetPixelIDValue())

        ## Resample after having transformed the moving image manually (inverse transform)
        moving_rigidly_warped_sitkh_sitk = sitkh.get_transformed_sitk_image(moving.sitk, sitk.Euler3DTransform(rigid_transform_sitk.GetInverse()))
        moving_rigidly_warped_manual_sitk = sitk.Resample(moving_rigidly_warped_sitkh_sitk, fixed.sitk, sitk.Euler3DTransform(), interpolator, default_pixel_value, moving.sitk.GetPixelIDValue())
        
        ## Compute difference image
        moving_rigidly_warped_diff_sitk = moving_rigidly_warped_sitk - moving_rigidly_warped_manual_sitk
        nda_diff_rigid = sitk.GetArrayFromImage(moving_rigidly_warped_diff_sitk)

        ## Check alignment
        self.assertEqual(np.round( np.linalg.norm(nda_diff_rigid), decimals = self.accuracy), 0)

        ## 2) Affine case
        ## Resample based on obtained registration trafo
        moving_affinely_warped_sitk = sitk.Resample(moving.sitk, fixed.sitk, affine_transform_sitk, interpolator, default_pixel_value, moving.sitk.GetPixelIDValue())

        ## Resample after having transformed the moving image manually (inverse transform)
        moving_affinely_warped_sitkh_sitk = sitkh.get_transformed_sitk_image(moving.sitk, sitk.AffineTransform(affine_transform_sitk.GetInverse()))
        moving_affinely_warped_manual_sitk = sitk.Resample(moving_affinely_warped_sitkh_sitk, fixed.sitk, sitk.Euler3DTransform(), interpolator, default_pixel_value, moving.sitk.GetPixelIDValue())

        ## Compute difference image
        moving_affinely_warped_diff_sitk = moving_affinely_warped_sitk - moving_affinely_warped_manual_sitk
        nda_diff_affine = sitk.GetArrayFromImage(moving_affinely_warped_diff_sitk)

        ## Check alignment
        self.assertEqual(np.round( np.linalg.norm(nda_diff_affine), decimals = self.accuracy), 0)

        ## Show some inermediate results
        # sitkh.show_sitk_image(fixed.sitk, overlay=moving_rigidly_warped_sitk, title="fixed_moving_rigid")
        # sitkh.show_sitk_image(fixed.sitk, overlay=moving_affinely_warped_sitk, title="fixed_moving_affine")
        # sitkh.show_sitk_image(moving_rigidly_warped_sitk, overlay=moving_affinely_warped_sitk, title="rigid_affine")


    def test_get_indices_array_to_flattened_sitk_image(self):
        # filename = "fetal_brain_0"
        filename = "FetalBrain_reconstruction_3stacks_myAlg"

        ## 3D
        image_sitk = sitk.ReadImage(self.dir_test_data + filename + ".nii.gz")
        nda = sitk.GetArrayFromImage(image_sitk).flatten()
        
        indices = sitkh.get_indices_array_to_flattened_sitk_image_data_array(image_sitk)
        nda_2 = np.zeros_like(nda)
        for i in range(0,nda_2.size):
            nda_2[i] = image_sitk.GetPixel(*indices[:,i])
        
        self.assertEqual(np.round( 
            np.linalg.norm(nda_2 - nda), 
        decimals = self.accuracy), 0)

        ## 2D
        image_sitk = image_sitk[:,:,5]
        nda = sitk.GetArrayFromImage(image_sitk).flatten()
        
        indices = sitkh.get_indices_array_to_flattened_sitk_image_data_array(image_sitk)
        nda_2 = np.zeros_like(nda)
        for i in range(0,nda_2.size):
            nda_2[i] = image_sitk.GetPixel(*indices[:,i])
        
        self.assertEqual(np.round( 
            np.linalg.norm(nda_2 - nda), 
        decimals = self.accuracy), 0)
        


# nda_rigid = sitk.GetArrayFromImage(moving_rigidly_warped_sitk)
# nda_rigid_manual = sitk.GetArrayFromImage(moving_rigidly_warped_manual_sitk)
# norm_diff_rigid = np.linalg.norm(nda_rigid-nda_rigid_manual)

# nda_affine = sitk.GetArrayFromImage(moving_affinely_warped_sitk)
# nda_affine_manual = sitk.GetArrayFromImage(moving_affinely_warped_manual_sitk)
# norm_diff_affine = np.linalg.norm(nda_affine-nda_affine_manual)

# print("norm_diff_rigid = " + str(norm_diff_rigid))
# print("norm_diff_affine = " + str(norm_diff_affine))




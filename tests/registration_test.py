# \file TestRegistration.py
#  \brief  Class containing unit tests for module Stack
#
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2016


# Import libraries
import SimpleITK as sitk
import itk
import numpy as np
import unittest
import os
import sys
import time
from datetime import timedelta

import pysitk.simple_itk_helper as sitkh
import pysitk.python_helper as ph

# Import modules
import niftymic.base.stack as st
import niftymic.registration.registration as myreg
import niftymic.registration.simple_itk_registration as regsitk
import niftymic.registration.cpp_itk_registration as regitk

from niftymic.definitions import DIR_TEST


# Pixel type of used 3D ITK image
PIXEL_TYPE = itk.D

# ITK image type
IMAGE_TYPE = itk.Image[PIXEL_TYPE, 3]
IMAGE_TYPE_CV33 = itk.Image.CVD33
IMAGE_TYPE_CV183 = itk.Image.CVD183
IMAGE_TYPE_CV363 = itk.Image.CVD363


class RegistrationTest(unittest.TestCase):

    # Specify input data
    dir_test_data = os.path.join(DIR_TEST, "registration/")

    accuracy = 2

    def setUp(self):
        # Set print options for numpy
        np.set_printoptions(precision=3)

    """
    def test_GradientEuler3DTransformImageFilter(self):

        filename_HRVolume = "HRVolume"

        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HRVolume + ".nii.gz")
        )

        DOF_transform = 6
        parameters = np.random.rand(
            DOF_transform)*(2*np.pi, 2*np.pi, 2*np.pi, 10, 10, 10)

        itk2np = itk.PyBuffer[IMAGE_TYPE]
        itk2np_CVD33 = itk.PyBuffer[IMAGE_TYPE_CV33]
        itk2np_CVD183 = itk.PyBuffer[IMAGE_TYPE_CV183]

        # Create Euler3DTransform and update with random parameters
        transform_itk = itk.Euler3DTransform.New()
        parameters_itk = transform_itk.GetParameters()
        sitkh.update_itk_parameters(parameters_itk, parameters)
        transform_itk.SetParameters(parameters_itk)
        # sitkh.print_itk_array(parameters_itk)

        # ---------------------------------------------------------------------
        filter_gradient_transform = itk.GradientEuler3DTransformImageFilter[
            IMAGE_TYPE, PIXEL_TYPE, PIXEL_TYPE].New()
        filter_gradient_transform.SetInput(HR_volume.itk)
        filter_gradient_transform.SetTransform(transform_itk)
        time_start = ph.start_timing()  # Above is required only once in Registration

        filter_gradient_transform.Update()
        gradient_transform_itk = filter_gradient_transform.GetOutput()
        # Get data array of Jacobian of transform w.r.t. parameters  and
        # reshape to N_HR_volume_voxels x DIMENSION x DOF
        nda_gradient_transform_1 = itk2np_CVD183.GetArrayFromImage(
            gradient_transform_itk).reshape(-1, 3, DOF_transform)

        print("GradientEuler3DTransformImageFilter: " +
              str(ph.stop_timing(time_start)))

        # ---------------------------------------------------------------------
        time_start = ph.start_timing()

        nda_gradient_transform_2 = sitkh.get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image(
            transform_itk, HR_volume.sitk)

        print("get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image: " +
              str(ph.stop_timing(time_start)))

        # ---------------------------------------------------------------------
        self.assertEqual(np.round(
            np.linalg.norm(nda_gradient_transform_2-nda_gradient_transform_1), decimals=6), 0)

    def test_GradientAffine3DTransformImageFilter(self):

        filename_HRVolume = "HRVolume"
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HRVolume + ".nii.gz")
        )

        DOF_transform = 12
        parameters = np.random.rand(DOF_transform)*10

        itk2np = itk.PyBuffer[IMAGE_TYPE]
        itk2np_CVD33 = itk.PyBuffer[IMAGE_TYPE_CV33]
        itk2np_CVD363 = itk.PyBuffer[IMAGE_TYPE_CV363]

        # Create Euler3DTransform and update with random parameters
        transform_itk = itk.AffineTransform[PIXEL_TYPE, 3].New()
        parameters_itk = transform_itk.GetParameters()
        sitkh.update_itk_parameters(parameters_itk, parameters)
        transform_itk.SetParameters(parameters_itk)
        # sitkh.print_itk_array(parameters_itk)

        # ---------------------------------------------------------------------
        filter_gradient_transform = itk.GradientAffine3DTransformImageFilter[
            IMAGE_TYPE, PIXEL_TYPE, PIXEL_TYPE].New()
        filter_gradient_transform.SetInput(HR_volume.itk)
        filter_gradient_transform.SetTransform(transform_itk)
        time_start = ph.start_timing()  # Above is required only once in Registration

        filter_gradient_transform.Update()
        gradient_transform_itk = filter_gradient_transform.GetOutput()
        # Get data array of Jacobian of transform w.r.t. parameters  and
        # reshape to N_HR_volume_voxels x DIMENSION x DOF
        nda_gradient_transform_1 = itk2np_CVD363.GetArrayFromImage(
            gradient_transform_itk).reshape(-1, 3, DOF_transform)

        print("GradientAffine3DTransformImageFilter: " +
              str(ph.stop_timing(time_start)))

        # ---------------------------------------------------------------------
        time_start = ph.start_timing()

        nda_gradient_transform_2 = sitkh.get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image(
            transform_itk, HR_volume.sitk)

        print("get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image: " +
              str(ph.stop_timing(time_start)))

        # ---------------------------------------------------------------------
        self.assertEqual(np.round(
            np.linalg.norm(nda_gradient_transform_2-nda_gradient_transform_1), decimals=6), 0)

    def test_reshaping_of_structures(self):

        # filename_prefix = "RigidTransform_"
        filename_prefix = "TranslationOnly_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"

        stack_sim = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_StackSim + ".nii.gz")
        )
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HRVolume + ".nii.gz")
        )

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        itk2np = itk.PyBuffer[itk.Image.D3]
        itk2np_CVD33 = itk.PyBuffer[itk.Image.CVD33]

        filter_OrientedGaussian_3D = itk.OrientedGaussianInterpolateImageFilter[
            IMAGE_TYPE, IMAGE_TYPE].New()
        filter_OrientedGaussian_3D.SetInput(HR_volume.itk)
        filter_OrientedGaussian_3D.SetUseJacobian(True)

        for j in range(0, N_slices):
            slice = slices_sim[j]

            filter_OrientedGaussian_3D.SetOutputParametersFromImage(slice.itk)
            filter_OrientedGaussian_3D.UpdateLargestPossibleRegion()
            filter_OrientedGaussian_3D.Update()

            slice_simulated_nda = itk2np.GetArrayFromImage(
                filter_OrientedGaussian_3D.GetOutput())
            dslice_simulated_nda = itk2np_CVD33.GetArrayFromImage(
                filter_OrientedGaussian_3D.GetJacobian())

            shape = slice_simulated_nda.shape

            slice_simulated_nda_flat = slice_simulated_nda.flatten()
            dslice_simulated_nda_flat = dslice_simulated_nda.reshape(-1, 3)

            array0 = np.zeros(3)
            array1 = np.zeros(3)
            abs_diff = 0

            iter = 0
            for i in range(0, shape[0]):
                for j in range(0, shape[1]):
                    for k in range(0, shape[2]):
                        array0 = slice_simulated_nda[
                            i, j, k] - dslice_simulated_nda[i, j, k, :]
                        array1 = slice_simulated_nda_flat[
                            iter] - dslice_simulated_nda_flat[iter, :]
                        abs_diff += np.linalg.norm(array0-array1)
                        iter += 1

            self.assertEqual(np.round(
                abs_diff, decimals=self.accuracy), 0)

    def test_vectorization_of_dImage_times_dT(self):

        shape = np.array([200, 200])
        slice_nda = np.random.rand(shape[0], shape[1])*255

        slice_sitk = sitk.GetImageFromArray(slice_nda)
        N_voxels = shape.prod()

        gradient_image_filter_sitk = sitk.GradientImageFilter()
        dslice_sitk = gradient_image_filter_sitk.Execute(slice_sitk)

        # Reshape to (N_slice_voxels x dim)-array
        dslice_nda = sitk.GetArrayFromImage(dslice_sitk).reshape(-1, 2)

        euler_sitk = sitk.Euler2DTransform()
        euler_itk = itk.Euler2DTransform.New()

        parameters_sitk = (0.5, -4, 10)
        euler_sitk.SetParameters(parameters_sitk)
        euler_itk.SetParameters(itk.OptimizerParameters[
                                itk.D](parameters_sitk))

        # Get d[T(theta, x)]/dtheta as (N_slice_voxels x dim x
        # transform_type_dofs)
        dT_nda = sitkh.get_numpy_array_of_jacobian_itk_transform_applied_on_sitk_image(
            euler_itk, slice_sitk)

        jacobian = np.zeros((N_voxels, euler_itk.GetNumberOfParameters()))

        time0 = ph.start_timing()
        for i in range(0, N_voxels):
            jacobian[i, :] = dslice_nda[i, :].dot(dT_nda[i, :, :])
        print("For-loop:   " + str(ph.stop_timing(time0)))

        time0 = ph.start_timing()
        jacobian_2 = np.sum((dslice_nda[:, :, np.newaxis]*dT_nda), axis=1)
        print("Vectorized: " + str(ph.stop_timing(time0)))

        self.assertEqual(np.round(
            np.linalg.norm(jacobian - jacobian_2), decimals=8), 0)

    def test_inplane_similarity_registration(self):
        filename = "HRVolume"
        scale = 0.9
        parameters = np.array([0.01, -0.05, 0.02, 2, -5, 10])

        fixed = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename + ".nii.gz"))

        # Generate test case:
        motion_sitk = sitk.VersorRigid3DTransform()
        motion_sitk.SetParameters(parameters)
        motion_matrix = np.array(motion_sitk.GetMatrix()).reshape(3, 3)

        # Alter image
        image_sitk = sitk.Image(fixed.sitk)
        spacing = np.array(image_sitk.GetSpacing())
        direction = np.array(image_sitk.GetDirection()).reshape(3, 3)
        direction_inv = np.linalg.inv(direction)
        origin = np.array(image_sitk.GetOrigin())

        # Alter spacing
        spacing[0:-1] *= scale
        Lambda = np.eye(3)
        Lambda[0:-1, :] *= scale
        image_sitk.SetSpacing(spacing)

        # Alter direction
        image_sitk.SetDirection((motion_matrix.dot(direction)).flatten())

        # Alter origin
        image_sitk.SetOrigin(motion_matrix.dot(direction).dot(Lambda).dot(
            direction_inv).dot(origin) + motion_sitk.GetTranslation())

        moving = st.Stack.from_sitk_image(image_sitk, filename + "_corrupted")

        # Perform registration
        registration = regitk.CppItkRegistration(fixed=fixed, moving=moving)
        registration.set_registration_type("InplaneSimilarity")
        registration.set_interpolator("NearestNeighbor")
        registration.set_metric("Correlation")
        # registration.set_scales_estimator("PhysicalShift")
        # registration.use_verbose(True)
        registration.run()

        parameters_estimated = registration.get_parameters()
        fixed_parameters_estimated = registration.get_parameters_fixed()

        # print("parameters_estimated = " + str(parameters_estimated))
        # print("fixed_parameters_estimated = " + str(fixed_parameters_estimated))

        # Extract information from in-plane similarity transform
        scale_estimated = parameters_estimated[-1]
        versors_estimated = parameters_estimated[0:3]
        translation_estimated = parameters_estimated[3:6]
        center_estimated = fixed_parameters_estimated[0:3]

        print("\t|scale - scale_estimated| = " +
              str(abs(scale-scale_estimated)))
        print("\t|rotation - rotation_estimated| (versors) = " +
              str(abs(parameters[0:3]-parameters_estimated[0:3])))
        print("\t|translation - translation_estimated| (versors) = " +
              str(abs(parameters[3:6]-parameters_estimated[3:6])))

        self.assertEqual(np.round(
            abs(scale-scale_estimated), decimals=self.accuracy), 0)
        self.assertEqual(np.round(
            np.linalg.norm(parameters[0:3]-parameters_estimated[0:3]), decimals=self.accuracy), 0)
        self.assertEqual(np.round(
            np.linalg.norm(parameters[3:6]-parameters_estimated[3:6]), decimals=0), 0)

        # Test translation to standard transforms:

        # Create scaling matrix Lambda
        Lambda_estimated = np.eye(3)
        Lambda_estimated[0:-1, :] *= scale_estimated

        # Result based on VersorRigid3DTransform (parameter information is
        # given as versor)
        rigid_sitk = sitk.VersorRigid3DTransform()
        rigid_sitk.SetParameters(parameters_estimated)
        R = np.array(rigid_sitk.GetMatrix()).reshape(3, 3)
        rigid_sitk.SetTranslation(R.dot(direction).dot(Lambda_estimated).dot(direction_inv).dot(
            origin-center_estimated) - R.dot(origin) + translation_estimated + center_estimated)
        fixed_scaled_sitk = sitk.Image(fixed.sitk)
        fixed_scaled_sitk.SetSpacing(spacing)
        moving_recovered_rigid = sitk.Resample(
            moving.sitk, fixed_scaled_sitk, rigid_sitk, sitk.sitkNearestNeighbor)
        # sitkh.show_sitk_image([fixed_scaled_sitk, moving_recovered_rigid], [filename, filename+"_recovered_rigid_no_scaling"])
        nda_diff = sitk.GetArrayFromImage(
            moving_recovered_rigid-fixed_scaled_sitk)
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

        # Result based on AffineTransform accounting for scaling too
        affine_matrix = R.dot(direction).dot(
            Lambda_estimated).dot(direction_inv)
        affine_sitk = sitk.AffineTransform(3)
        affine_sitk.SetMatrix(affine_matrix.flatten())
        affine_sitk.SetCenter(center_estimated)
        affine_sitk.SetTranslation(translation_estimated)
        moving_recovered_affine = sitk.Resample(
            moving.sitk, fixed.sitk, affine_sitk, sitk.sitkNearestNeighbor)
        # sitkh.show_sitk_image([fixed.sitk, moving_recovered_affine], [filename, filename+"_recovered_affine"])
        nda_diff = sitk.GetArrayFromImage(moving_recovered_affine-fixed.sitk)
        self.assertEqual(np.round(
            np.linalg.norm(nda_diff), decimals=self.accuracy), 0)

        # Alternatively: Create Euler 3D Transform (by converting quaternion to euler coordinates)
        # Problem is: I can't figure the right conversion it seems ...
        # q = np.zeros(4)
        # q[0:3] = np.array(registration.get_parameters()[0:3])
        # q[3] = np.sqrt(1-q.dot(q))
        # alpha = np.arctan2(2*(q[0]*q[1]+q[2]*q[3]),1-2*(q[1]**2+q[2]**2))
        # beta = np.arcsin(2*(q[0]*q[2]-q[3]*q[1]))
        # gamma = np.arctan2(2*(q[0]*q[3]+q[1]*q[2]),1-2*(q[2]**2+q[3]**2))
        # euler_sitk = sitk.Euler3DTransform()
        # R_euler = np.array(euler_sitk.GetMatrix()).reshape(3,3)
        # euler_sitk.SetRotation(alpha, beta, gamma)
        # euler_sitk.SetTranslation(R.dot(direction).dot(Lambda_estimated).dot(direction_inv).dot(origin-center_estimated) - R.dot(origin) + translation_estimated + center_estimated)
        # moving_recovered_euler = sitk.Resample(moving.sitk, fixed_scaled_sitk, euler_sitk, sitk.sitkNearestNeighbor)
        # # sitkh.show_sitk_image([fixed_scaled_sitk, moving_recovered_euler], [filename, filename+"_recovered_euler_no_scaling"])

        # for i in range(0, 2):
        #     for j in range(0, 2):
        #         for k in range(0, 2):
        #             print("(i,j,k)=(%s,%s,%s)" %(i,j,k))
        #             foo_min = 100
        #             foo_index = 100
        #             for l in range(0, 6):
        #                 euler_sitk = sitk.Euler3DTransform()
        #                 if l is 0:
        #                     euler_sitk.SetRotation(alpha*(-1)**i, beta*(-1)**j, gamma*(-1)**k)
        #                 if l is 1:
        #                     euler_sitk.SetRotation(alpha*(-1)**i, gamma*(-1)**j, beta*(-1)**k)
        #                 if l is 2:
        #                     euler_sitk.SetRotation(beta*(-1)**i, gamma*(-1)**j, alpha*(-1)**k)
        #                 if l is 3:
        #                     euler_sitk.SetRotation(beta*(-1)**i, alpha*(-1)**j, gamma*(-1)**k)
        #                 if l is 4:
        #                     euler_sitk.SetRotation(gamma*(-1)**i, alpha*(-1)**j, beta*(-1)**k)
        #                 if l is 5:
        #                     euler_sitk.SetRotation(gamma*(-1)**i, beta*(-1)**j, alpha*(-1)**k)

        #                 R_euler = np.array(euler_sitk.GetMatrix()).reshape(3,3)
        #                 # euler_sitk.SetComputeZYX(True)
        #                 diff = np.linalg.norm(R-R_euler)
        #                 if diff < foo_min:
        #                     foo = diff;
        #                     foo_index = l
        #                     # print(euler_sitk.GetParameters())
        #                 # print(diff)
        #             print("Minimum = %s at index = %s" %(foo, foo_index))

    """

    def test_translation_registration_of_slices(self):

        filename_prefix = "TranslationOnly_"
        # filename_prefix = "RigidTransform_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"
        stack_sim = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_StackSim + ".nii.gz"))
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HRVolume + ".nii.gz"))

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        scale = np.array(
            [180. / np.pi, 180. / np.pi, 180. / np.pi, 1., 1., 1.])
        time_start = ph.start_timing()

        for j in range(0, N_slices):
            # for j in range(20, N_slices):
            rigid_transform_groundtruth_sitk = sitk.ReadTransform(
                self.dir_test_data + filename_transforms_prefix + str(j) + ".tfm")
            parameters_gd = np.array(
                rigid_transform_groundtruth_sitk.GetParameters())

            angle_max = 5. * np.pi / 180.
            t_max = 5.
            registration = myreg.Registration(
                fixed=slices_sim[j], moving=HR_volume,
                # initializer_type="SelfGEOMETRY",
                use_verbose=0,
                # data_loss="soft_l1",
                # x_scale=[angle_max, angle_max, angle_max, t_max, t_max, t_max],
            )
            registration.run()
            # registration.print_statistics()

            # Check parameters
            transform_sitk = registration.get_registration_transform_sitk()
            parameters = np.array(transform_sitk.GetParameters())
            norm_diff = np.linalg.norm(parameters-parameters_gd)

            params = parameters * scale
            params_gt = parameters_gd * scale
            print("Slice %s/%s: |parameters-parameters_gd| = %s" %
                  (j, N_slices-1, str(norm_diff)))
            print("\tEst:  " + str(params) + " (deg/mmm)")
            print("\tGT:   " + str(params_gt) + " (deg/mmm)")
            print("\tDiff: " + str(params - params_gt) + " (deg/mmm)")

            self.assertEqual(np.round(
                norm_diff, decimals=self.accuracy), 0)

        # Set elapsed time
        print("Translation: " + str(ph.stop_timing(time_start)))

    """
    def test_rigid_registration_of_slices(self):

        filename_prefix = "RigidTransform_"

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        filename_transforms_prefix = filename_prefix + "TransformGroundTruth_slice"
        stack_sim = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_StackSim + ".nii.gz"))
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HRVolume + ".nii.gz"))

        slices_sim = stack_sim.get_slices()
        N_slices = len(slices_sim)

        time_start = time.time()

        for j in range(0, N_slices):
            rigid_transform_groundtruth_sitk = sitk.ReadTransform(
                self.dir_test_data + filename_transforms_prefix + str(j) + ".tfm")
            parameters_gd = np.array(
                rigid_transform_groundtruth_sitk.GetParameters())

            registration = myreg.Registration(
                fixed=slices_sim[j], moving=HR_volume)
            registration.run()
            # registration.print_statistics()

            # Check parameters
            parameters = registration.get_parameters()

            norm_diff = np.linalg.norm(parameters-parameters_gd)
            # print("Slice %s/%s: |parameters-parameters_gd| = %s" %(j, N_slices-1, str(norm_diff)) )

            self.assertEqual(np.round(
                norm_diff, decimals=self.accuracy), 0)

        # Set elapsed time
        time_end = time.time()
        # print("Rigid registration test: Elapsed time = %s" %(timedelta(seconds=time_end-time_start)))

    def test_rigid_registration_of_stack(self):
        filename_prefix = "NoMotion_"
        parameters_gd = (0.1, 0.1, 0.2, -1, 3, 2)

        filename_HRVolume = "HRVolume"
        filename_StackSim = filename_prefix + "StackSimulated"
        stack_sim = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_StackSim + ".nii.gz"))
        HR_volume = st.Stack.from_filename(
            os.path.join(self.dir_test_data, filename_HRVolume + ".nii.gz"))

        # Apply motion
        transform_sitk = sitk.Euler3DTransform()
        transform_sitk.SetParameters(parameters_gd)
        stack_sitk = sitkh.get_transformed_sitk_image(
            stack_sim.sitk, transform_sitk)
        stack_sitk_mask = sitkh.get_transformed_sitk_image(
            stack_sim.sitk_mask, transform_sitk)
        stack_sim = st.Stack.from_sitk_image(
            stack_sitk, filename=stack_sim.get_filename(), image_sitk_mask=stack_sitk_mask)

        # PSF-aware Registration algorithm
        time_start = time.time()
        registration = myreg.Registration(
            fixed=stack_sim,
            moving=HR_volume,
            use_verbose=True,
            # data_loss="huber",
            # minimizer="L-BFGS-B",
            # minimizer="Newton-CG",
        )
        # registration.use_fixed_mask(True)
        registration.use_verbose(True)
        registration.run()

        # Check parameters (should be the negative of parameters_gd)
        parameters = registration.get_parameters()
        norm_diff = np.linalg.norm(parameters+parameters_gd)
        print("parameters = " + str(parameters))
        print("|parameters-parameters_gd| = %s" % (str(norm_diff)))

        self.assertEqual(np.round(
            norm_diff*0.1, decimals=0), 0)

        # Set elapsed time
        time_end = time.time()
        print("Rigid registration test: Elapsed time = %s" %
              (registration.get_computational_time()))

        # Comparison with ITK registrations
        scales_estimator = "PhysicalShift"
        use_verbose = 1

        # ----------------SimpleITK registration for comparison----------------
        ph.print_title("SimpleITK registration for comparison:")
        registration = regsitk.SimpleItkRegistration(
            fixed=stack_sim,
            moving=HR_volume,
            optimizer="RegularStepGradientDescent",
            optimizer_params={
                "minStep": 1e-6,
                "numberOfIterations": 500,
                "gradientMagnitudeTolerance": 1e-6,
                "learningRate": 1,
            },
        )
        registration.set_metric("MeanSquares")
        registration.use_verbose(use_verbose)
        registration.set_scales_estimator(scales_estimator)
        registration.run()

        parameters = np.array(
            registration.get_registration_transform_sitk().GetParameters())
        norm_diff = np.linalg.norm(parameters+parameters_gd)
        print("\tparameters = " + str(parameters))
        print("\t|parameters-parameters_gd| = %s" % (str(norm_diff)))
        print("\tRigid registration test: Elapsed time = %s" %
              (registration.get_computational_time()))

        # -------------------ITK registration for comparison-------------------
        ph.print_title("ITK registration for comparison:")
        registration = regitk.CppItkRegistration(
            fixed=stack_sim, moving=HR_volume)
        registration.use_verbose(use_verbose)
        registration.set_scales_estimator(scales_estimator)
        registration.set_metric("MeanSquares")
        registration.run()

        parameters = np.array(
            registration.get_registration_transform_sitk().GetParameters())
        norm_diff = np.linalg.norm(parameters+parameters_gd)
        print("\tparameters = " + str(parameters))
        print("\t|parameters-parameters_gd| = %s" % (str(norm_diff)))
        print("\tRigid registration test: Elapsed time = %s" %
              (registration.get_computational_time()))
    """

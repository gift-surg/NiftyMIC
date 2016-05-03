## \file InPlaneRigidRegistration.py
#  \brief  
# 
#  \author Michael Ebner (michael.ebner.14@ucl.ac.uk)
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import sys
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

## Import modules from src-folder
import SimpleITKHelper as sitkh
import Stack as st


class InPlaneRigidRegistration:

    def __init__(self, stack_manager):

        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        self._stacks_of_planarly_aligned_slices = [None]*len(self._stacks)
        # self._resampled_sitk_stacks_after_in_plane_alignment = [None]*len(self._stacks)
        # self._resampled_sitk_stack_masks_after_in_plane_alignment = [None]*len(self._stacks)

        self._UT_2D_resampled_stacks = [None]*len(self._stacks)


        return None


    def run_in_plane_rigid_registration(self):

        ## When running unit test
        # self._UT_apply_in_plane_rigid_registration_2D_approach_01()


        ## Apply in-plane registration on each stack separately
        for i in range(0, self._N_stacks):

            # self._apply_in_plane_rigid_registration_2D_approach_01(self._stacks[i])
            self._apply_in_plane_rigid_registration_2D_approach_02(self._stacks[i])
            # self._apply_in_plane_rigid_registration_2D_approach_02_Alternative(self._stacks[i])

        return None


    ## Simply register one slice to the previous one
    def _apply_in_plane_rigid_registration_2D_approach_01(self, stack):
        slices = stack.get_slices()
        N_slices = stack.get_number_of_slices()

        ## 
        step = 1

        for iterations in range(0,1):
            for j in range(0, N_slices-step):

                ## Aim: register slice_3D to slice_3D_ref in-plane
                slice_3D = slices[j+step]
                slice_3D_ref = slices[j]

                ## Get copy of slices aligned to physical coordinate system
                slice_3D_copy_sitk = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(slice_3D)
                slice_3D_ref_copy_sitk = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(slice_3D_ref)


                if slices[j].sitk_mask is None:
                    ## Get registration trafo for slices in physical 2D space (moving_2D -> fixed_2D)
                    rigid_transform_2D_inv = self._in_plane_rigid_registration_2D(
                        fixed_2D_sitk = slice_3D_ref_copy_sitk[:,:,0], 
                        moving_2D_sitk = slice_3D_copy_sitk[:,:,0])
                         # fixed_2D_sitk_mask=None, moving_2D_sitk_mask=None)

                ## Fetch and update masks if existing:
                else:
                    ## Mask information of fixed
                    mask_nda = sitk.GetArrayFromImage(slice_3D_ref.sitk_mask)
                    slice_3D_ref_copy_sitk_mask = sitk.GetImageFromArray(mask_nda)
                    slice_3D_ref_copy_sitk_mask.CopyInformation(slice_3D_ref_copy_sitk)

                    ## Mask information of moving
                    mask_nda = sitk.GetArrayFromImage(slice_3D.sitk_mask)
                    slice_3D_copy_sitk_mask = sitk.GetImageFromArray(mask_nda)
                    slice_3D_copy_sitk_mask.CopyInformation(slice_3D_copy_sitk)

                    ## Get registration trafo for slices in physical 2D space (moving_2D -> fixed_2D)
                    rigid_transform_2D_inv = self._in_plane_rigid_registration_2D(
                        fixed_2D_sitk = slice_3D_ref_copy_sitk[:,:,0], 
                        moving_2D_sitk = slice_3D_copy_sitk[:,:,0],
                        fixed_2D_sitk_mask = slice_3D_ref_copy_sitk_mask[:,:,0], 
                        moving_2D_sitk_mask = slice_3D_copy_sitk_mask[:,:,0])


                ## Get transformation for 3D in-plane rigid transformation to update T_PI of slice
                T_PP = self._get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D.sitk)

                ## Get registration trafo for slices in physical 3D space
                T_PI_in_plane_rotation_3D = self._get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update T_PI of slice s.t. it is aligned with slice_3D_ref
                slice_3D.update_affine_transform(T_PI_in_plane_rotation_3D)

        return None


    def _get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(self, rigid_transform_2D_sitk, T, slice_3D_sitk):
        ## Extract affine transformation to transform from Image to Physical Space
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_3D_sitk)

        ## T = T_rotation_inv o T_origin_inv
        # T = get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D)
        # T = self._get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D.sitk)
        T_inv = sitk.AffineTransform(T.GetInverse())

        ## T_PI_align = T_rotation_inv o T_origin_inv o T_PI: Trafo to align stack with physical coordinate system
        ## (Hence, T_PI_align(\i) = \spacing*\i)
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T, T_PI)

        ## Extract direction matrix and origin so that slice is oriented according to T_PI_align (i.e. with physical axes)
        # origin_PI_align = get_sitk_image_origin_from_sitk_affine_transform(T_PI_align,slice_3D)
        # direction_PI_align = get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align,slice_3D)

        ## Extend to 3D rigid transform
        rigid_transform_3D = self._get_3D_from_sitk_2D_rigid_transform(rigid_transform_2D_sitk) 

        ## T_PI_in_plane_rotation_3D 
        ##    = T_origin o T_rotation o T_in_plane_rotation_2D_space 
        ##                      o T_rotation_inv o T_origin_inv o T_PI
        T_PI_in_plane_rotation_3D = sitk.AffineTransform(3)
        T_PI_in_plane_rotation_3D = sitkh.get_composited_sitk_affine_transform(rigid_transform_3D, T_PI_align)
        T_PI_in_plane_rotation_3D = sitkh.get_composited_sitk_affine_transform(T_inv, T_PI_in_plane_rotation_3D)

        return T_PI_in_plane_rotation_3D


    def _get_3D_from_sitk_2D_rigid_transform(self, rigid_transform_2D_sitk):
        # Get parameters of 2D registration
        angle_z, translation_x, translation_y = rigid_transform_2D_sitk.GetParameters()
        center_x, center_y = rigid_transform_2D_sitk.GetFixedParameters()

        # Expand obtained translation to 3D vector
        translation_3D = (translation_x, translation_y, 0)
        center_3D = (center_x, center_y, 0)

        # Create 3D rigid transform based on 2D
        rigid_transform_3D = sitk.Euler3DTransform()
        rigid_transform_3D.SetRotation(0,0,angle_z)
        rigid_transform_3D.SetTranslation(translation_3D)
        rigid_transform_3D.SetFixedParameters(center_3D)
        
        return rigid_transform_3D


    def _check_sitk_mask_2D(self, mask_2D_sitk):

        mask_nda = sitk.GetArrayFromImage(mask_2D_sitk)

        if np.sum(mask_nda) > 1:
            return mask_2D_sitk

        else:
            mask_nda[:] = 1

            mask = sitk.GetImageFromArray(mask_nda)
            mask.CopyInformation(mask_2D_sitk)

            return mask


    def _apply_in_plane_rigid_registration_2D_approach_02(self, stack):
        slices = stack.get_slices()
        N_slices = stack.get_number_of_slices()

        iterations = 1
        for iteration in range(0,iterations):
            # for j in np.concatenate((range(0, N_slices, 2),range(1, N_slices, 2))):
            for j in np.concatenate((range(1, N_slices, 2),range(0, N_slices, 2))):
                slice_3D = slices[j]

                # print("Iteration %r/%r: Slice = %r/%r" %(iteration+1,iterations,j,N_slices-1))

                ## Get copy of slices aligned to physical coordinate system
                slice_3D_copy_sitk = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(slice_3D)

                ## Average neighbours of slices and retrieve slices aligned with physical coordinate system
                slice_2D_ref_sitk, slice_2D_ref_sitk_mask = self._get_average_of_slice_neighbours(slices, j)

                # if slices[j].sitk_mask is None:
                #     ## Get registration trafo for slices in physical 2D space (moving_2D -> fixed_2D)
                #     rigid_transform_2D_inv = self._in_plane_rigid_registration_2D(
                #         fixed_2D_sitk = slice_2D_ref_sitk, 
                #         moving_2D_sitk = slice_3D_copy_sitk[:,:,0])
                #          # fixed_2D_sitk_mask=None, moving_2D_sitk_mask=None)

                # ## Fetch and update masks if existing:
                # else:
                ## Mask information of moving
                slice_3D_copy_sitk_mask = sitk.Image(slice_3D.sitk_mask)
                slice_3D_copy_sitk_mask.SetOrigin(slice_3D_copy_sitk.GetOrigin())
                slice_3D_copy_sitk_mask.SetDirection(slice_3D_copy_sitk.GetDirection())

                ## Get registration trafo for slices in physical 2D space (moving_2D -> fixed_2D)
                rigid_transform_2D_inv = self._in_plane_rigid_registration_2D(
                    fixed_2D_sitk = slice_2D_ref_sitk, 
                    moving_2D_sitk = slice_3D_copy_sitk[:,:,0],
                    fixed_2D_sitk_mask = self._check_sitk_mask_2D(slice_2D_ref_sitk_mask), 
                    moving_2D_sitk_mask = self._check_sitk_mask_2D(slice_3D_copy_sitk_mask[:,:,0]))

                ## Get transformation for 3D in-plane rigid transformation to update T_PI of slice
                T_PP = self._get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D.sitk)

                ## Get registration trafo for slices in physical 3D space
                T_PI_in_plane_rotation_3D = self._get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update T_PI of slice s.t. it is aligned with slice_3D_ref
                slice_3D.update_affine_transform(T_PI_in_plane_rotation_3D)

        return None


    def _apply_in_plane_rigid_registration_2D_approach_02_Alternative(self, stack):
        slices = stack.get_slices()
        N_slices = stack.get_number_of_slices()

        methods = ["NiftyReg", "FLIRT"]
        method = methods[0]

        iterations = 1
        for iteration in range(0,iterations):
            # for j in np.concatenate((range(1, N_slices, 2),range(0, N_slices, 2))):
            for j in range(1, N_slices, 2):
            # for j in range(24, 25):
                slice_3D = slices[j]

                print("Iteration %r/%r: Slice = %r/%r" %(iteration+1,iterations,j,N_slices-1))

                ## Get copy of slices aligned to physical coordinate system
                slice_3D_copy_sitk = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(slice_3D)

                ## Average neighbours of slices and retrieve slices aligned with physical coordinate system
                slice_2D_ref_sitk, slice_2D_ref_sitk_mask = self._get_average_of_slice_neighbours(slices, j)

                ## Fetch and update masks if existing:
                slice_3D_copy_sitk_mask = sitk.Image(slice_3D.sitk_mask)
                slice_3D_copy_sitk_mask.SetOrigin(slice_3D_copy_sitk.GetOrigin())
                slice_3D_copy_sitk_mask.SetDirection(slice_3D_copy_sitk.GetDirection())

                ## Save images prior to the use of NiftyReg
                dir_tmp = ".tmp/" 
                os.system("mkdir -p " + dir_tmp)

                moving_sitk_2D = slice_3D_copy_sitk[:,:,0]
                moving_sitk_2D_mask = self._check_sitk_mask_2D(slice_3D_copy_sitk_mask[:,:,0])

                fixed_sitk_2D = slice_2D_ref_sitk
                fixed_sitk_2D_mask = self._check_sitk_mask_2D(slice_2D_ref_sitk_mask)

                moving_str = str(j) + "_moving" 
                moving_mask_str = str(j) +"_moving_mask"
                fixed_str = str(j) + "_fixed"
                fixed_mask_str = str(j) + "_fixed_mask"

                sitk.WriteImage(moving_sitk_2D, dir_tmp+moving_str+".nii.gz")
                sitk.WriteImage(moving_sitk_2D_mask, dir_tmp+moving_mask_str+".nii.gz")
                sitk.WriteImage(fixed_sitk_2D, dir_tmp+fixed_str+".nii.gz")
                sitk.WriteImage(fixed_sitk_2D_mask, dir_tmp+fixed_mask_str+".nii.gz")

                if method == "FLIRT":
                    res_affine_image = dir_tmp + moving_str + "_warped_FLIRT_" + str(j)
                    res_affine_matrix = dir_tmp + ".affine_matrix_FLIRT_" + str(j)

                    options = "-2D -cost mutualinfo "
                        # "-refweight " + fixed_mask_str + ".nii.gz " + \
                        # "-inweight " + moving_mask_str + ".nii.gz " + \

                    cmd = "flirt " + options + \
                        "-ref " + dir_tmp + fixed_str + ".nii.gz " + \
                        "-in " + dir_tmp + moving_str + ".nii.gz " + \
                        "-out " + dir_tmp + res_affine_image + ".nii.gz " + \
                        "-omat " + dir_tmp + res_affine_matrix + ".txt"
                    sys.stdout.write("  Rigid registration (FLIRT) " + str(j+1) + "/" + str(N_slices) + " ... ")


                else:
                    ## NiftyReg: Global affine registration of reference image:
                    #  \param[in] -ref reference image
                    #  \param[in] -flo floating image
                    #  \param[out] -res affine registration of floating image
                    #  \param[out] -aff affine transformation matrix
                    res_affine_image = moving_str + "_warped_NiftyReg"
                    res_affine_matrix = str(j) + "_affine_matrix_NiftyReg"

                    options = "-voff -rigOnly "
                    # options = "-voff -platf 1 "
                    cmd = "reg_aladin " + options + \
                        "-ref " + dir_tmp + fixed_str + ".nii.gz " + \
                        "-flo " + dir_tmp + moving_str + ".nii.gz " + \
                        "-rmask " + dir_tmp + fixed_mask_str + ".nii.gz " + \
                        "-fmask " + dir_tmp + moving_mask_str + ".nii.gz " + \
                        "-res " + dir_tmp + res_affine_image + ".nii.gz " + \
                        "-aff " + dir_tmp + res_affine_matrix + ".txt "
                    print(cmd)
                    sys.stdout.write("  Rigid registration (NiftyReg reg_aladin) " + str(j+1) + "/" + str(N_slices) + " ... ")

                sys.stdout.flush() #flush output; otherwise sys.stdout.write would wait until next newline before printing
                os.system(cmd)
                print "done"

                """
                ## Test within console
                # reg_aladin -rigOnly -ref .tmp/${N}_fixed.nii.gz -flo .tmp/${N}_moving.nii.gz -rmask .tmp/${N}_fixed_mask.nii.gz -fmask .tmp/${N}_moving_mask.nii.gz -res .tmp/${N}_moving_warped_NiftyReg.nii.gz -aff .tmp/${N}_affine_matrix_NiftyReg.txt; N=11;
                """

                ## Read trafo and invert such that T: moving_2D -> fixed_3D
                matrix = np.loadtxt(dir_tmp+res_affine_matrix+".txt")

                print matrix
                ## Convert to SimpleITK format:

                ## Negative representation of (x,y)-coordinates compared to nifti-header (cf. SimpleITK_PhysicalCoordinates.py) --> negative sign
                t_2D = -matrix[0:-2,-1]
                
                ## NiftyReg uses mathematically negative representation for rotation!! --> negative sign
                angle_z = -np.arccos(matrix[0,0])
                
                center_2D = (0,0)

                ## Obtain inverse translation
                tmp_trafo = sitk.Euler2DTransform((0,0),-angle_z,(0,0))
                t_2D_inv = tmp_trafo.TransformPoint(-t_2D)

                ## inverse = R_inv(x-c) - R_inv(t) + c
                rigid_transform_2D_inv = sitk.Euler2DTransform(center_2D, -angle_z, t_2D_inv)

                ## Get transformation for 3D in-plane rigid transformation to update T_PI of slice
                T_PP = self._get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D.sitk)

                ## Get registration trafo for slices in physical 3D space
                T_PI_in_plane_rotation_3D = self._get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update T_PI of slice s.t. it is aligned with slice_3D_ref
                slice_3D.update_affine_transform(T_PI_in_plane_rotation_3D)

                """
                """

                moving = slice_3D_copy_sitk[:,:,0]
                fixed = slice_2D_ref_sitk
                warped = sitk.ReadImage(dir_tmp+res_affine_image +".nii.gz",sitk.sitkFloat64)

                # print matrix

                # spacing = np.array(fixed.GetSpacing())
                # S = np.diag(spacing)
                # S_inv = np.diag(1/spacing)


                # A_test = transform[0:-2,0:-2]
                # t_test = transform[0:-2,-1]

                # dim = np.array(fixed.GetSize())
                # center_test = (dim/2.)

                # affine = sitk.AffineTransform(A_test.flatten(),t_test,center_test)

                # warped_sitk = sitk.Resample(moving, fixed, affine, sitk.sitkBSpline, 0.0, moving.GetPixelIDValue())


                # fig = plt.figure(1)
                # plt.suptitle(np.linalg.norm(sitk.GetArrayFromImage(fixed-warped)))
                # plt.subplot(1,3,1)
                # plt.imshow(sitk.GetArrayFromImage(fixed), cmap="Greys_r")
                # plt.axis('off')

                # plt.subplot(1,3,2)
                # plt.imshow(sitk.GetArrayFromImage(warped), cmap="Greys_r")
                # plt.axis('off')
                
                # plt.subplot(1,3,3)
                # plt.imshow(sitk.GetArrayFromImage(fixed-warped), cmap="Greys_r")
                # plt.axis('off')
                # plt.show()

                """
                """

        return None



    def _get_average_of_slice_neighbours(self, slices, slice_number):
        N_slices = len(slices)
        slice_3D = slices[slice_number]

        if slice_number == 0:
            slice_3D_next = slices[slice_number+1]

            average_3D_slice_sitk = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(
                slice_3D_next)

            average_3D_slice_sitk_mask = sitk.Image(slice_3D_next.sitk_mask)
            average_3D_slice_sitk_mask.SetOrigin(average_3D_slice_sitk.GetOrigin())
            average_3D_slice_sitk_mask.SetDirection(average_3D_slice_sitk.GetDirection())

            average_2D_slice_sitk = average_3D_slice_sitk[:,:,0]
            average_2D_slice_sitk_mask = average_3D_slice_sitk_mask[:,:,0]

        elif slice_number == N_slices-1:
            slice_3D_prev = slices[slice_number-1]

            average_3D_slice_sitk = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(
                slice_3D_prev)

            average_3D_slice_sitk_mask = sitk.Image(slice_3D_prev.sitk_mask)
            average_3D_slice_sitk_mask.SetOrigin(average_3D_slice_sitk.GetOrigin())
            average_3D_slice_sitk_mask.SetDirection(average_3D_slice_sitk.GetDirection())

            average_2D_slice_sitk = average_3D_slice_sitk[:,:,0]
            average_2D_slice_sitk_mask = average_3D_slice_sitk_mask[:,:,0]

        else:
            slice_3D_prev = slices[slice_number-1]
            slice_3D_next = slices[slice_number+1]

            ## Get 2D sitk slices aligned with physical coordinates
            # slice_2D_align = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(
            #     slice_3D)[:,:,0]
            slice_3D_prev_align = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(
                slice_3D_prev)
            slice_3D_next_align = self._get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(
                slice_3D_next)

            slice_2D_prev_align = slice_3D_prev_align[:,:,0]
            slice_2D_next_align = slice_3D_next_align[:,:,0]

            ## Resample previous 2D slice to slice_2D_prev space
            slice_2D_next_align_warped = sitk.Resample(slice_2D_next_align, slice_2D_prev_align, sitk.Euler2DTransform(), sitk.sitkLinear, 0.0, slice_2D_next_align.GetPixelIDValue())

            ## Masks:
            slice_3D_prev_align_mask = sitk.Image(slice_3D_prev.sitk_mask)
            slice_3D_prev_align_mask.SetOrigin(slice_3D_prev_align.GetOrigin())
            slice_3D_prev_align_mask.SetDirection(slice_3D_prev_align.GetDirection())
            slice_2D_prev_align_mask = slice_3D_prev_align_mask[:,:,0]

            slice_3D_next_align_mask = sitk.Image(slice_3D_next.sitk_mask)
            slice_3D_next_align_mask.SetOrigin(slice_3D_next_align.GetOrigin())
            slice_3D_next_align_mask.SetDirection(slice_3D_next_align.GetDirection())
            slice_2D_next_align_mask = slice_3D_next_align_mask[:,:,0]

            slice_2D_next_align_mask_warped = sitk.Resample(slice_2D_next_align_mask, slice_2D_prev_align_mask, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_next_align_mask.GetPixelIDValue())

            ## Compute average of images
            average_2D_slice_sitk =  ( slice_2D_prev_align + slice_2D_next_align_warped )/2.

            ## Compute average of masks
            average_2D_slice_sitk_mask_nda = np.round(
                ( sitk.GetArrayFromImage(slice_2D_prev_align_mask) + sitk.GetArrayFromImage(slice_2D_next_align_mask_warped) )/2.
                )
            average_2D_slice_sitk_mask = sitk.GetImageFromArray(average_2D_slice_sitk_mask_nda)
            average_2D_slice_sitk_mask.CopyInformation(slice_2D_prev_align_mask)

        return average_2D_slice_sitk, average_2D_slice_sitk_mask


    def _in_plane_rigid_registration_2D(self, fixed_2D_sitk, moving_2D_sitk, fixed_2D_sitk_mask, moving_2D_sitk_mask):
        ## Instantiate interface method to the modular ITKv4 registration framework
        registration_method = sitk.ImageRegistrationMethod()

        ## Select between using the geometrical center (GEOMETRY) of the images or using the center of mass (MOMENTS) given by the image intensities
        initial_transform = sitk.CenteredTransformInitializer(fixed_2D_sitk, moving_2D_sitk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
        # initial_transform = sitk.Euler2DTransform()

        ## Set the initial transform and parameters to optimize
        registration_method.SetInitialTransform(initial_transform)

        ## Set an image masks in order to restrict the sampled points for the metric
        # registration_method.SetMetricFixedMask(fixed_2D_sitk_mask)
        # registration_method.SetMetricMovingMask(moving_2D_sitk_mask)

        ## Set percentage of pixels sampled for metric evaluation
        # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

        ## Set interpolator to use
        registration_method.SetInterpolator(sitk.sitkLinear)

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
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)

        ## Use negative means squares image metric
        # registration_method.SetMetricAsMeanSquares()
        
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
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

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

        ## Execute 2D registration
        rigid_transform_2D = registration_method.Execute(sitk.Cast(fixed_2D_sitk, sitk.sitkFloat64), sitk.Cast(moving_2D_sitk, sitk.sitkFloat64)) 

        ## Return inverse transform
        return sitkh.get_inverse_of_sitk_rigid_registration_transform(rigid_transform_2D)


    def _get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(self, slice_3D):

        ## Create copies of fixed and moving slices
        slice_3D_sitk_copy = sitk.Image(slice_3D.sitk)

        ## Get current transform from image to physical space of slice
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_3D.sitk)

        ## Get transform to get alignment with physical coordinate system of original/untouched slice
        T_PP = self._get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D.sitk)

        ## Get transform to align slice with physical coordinate system (perhaps already shifted there) 
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T_PP, T_PI)

        ## Set direction and origin of image accordingly
        direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)
        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)

        slice_3D_sitk_copy.SetDirection(direction)
        slice_3D_sitk_copy.SetOrigin(origin)

        return slice_3D_sitk_copy



    def _get_3D_transform_to_align_stack_with_physical_coordinate_system(self, slice_3D_sitk):
        ## Extract origin and direction matrix from slice:
        origin_3D = np.array(slice_3D_sitk.GetOrigin())
        direction_3D = np.array(slice_3D_sitk.GetDirection())

        ## Generate inverse transformations for translation and orthogonal transformations
        T_translation = sitk.AffineTransform(3)
        T_translation.SetTranslation(-origin_3D)

        T_rotation = sitk.AffineTransform(3)
        direction_inv = np.linalg.inv(direction_3D.reshape(3,3)).flatten()
        T_rotation.SetMatrix(direction_inv)

        ## T = T_rotation_inv o T_origin_inv
        T = sitkh.get_composited_sitk_affine_transform(T_rotation,T_translation)

        return T


    ## Unit test: Simpy register one slice to the previous one
    def _UT_apply_in_plane_rigid_registration_2D_approach_01(self):
        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            ## Create stack aligned with physical coordinate system but with kept spacing information
            stack_sitk_tmp = sitk.Image(stack.sitk)
            stack_sitk_tmp.SetDirection(np.eye(3).flatten())
            stack_sitk_tmp.SetOrigin((0,0,0))

            ## * Unit Test add on
            UT_image = sitk.Image(stack.sitk)
            UT_image_planar = sitk.Image(UT_image)

            UT_image_nda = sitk.GetArrayFromImage(UT_image)
            ## * ## 

            step = 1

            for j in range(0, N_slices-step):

                # print("Iteration " + str(j+1) + "/" + str(N_slices-1) + ":")
                
                ## Choose slice which shall be registered
                slice_3D = slices[j+step]

                ## Choose fixed and moving slice for 2D registration
                fixed_2D_sitk = stack_sitk_tmp[:,:,j]
                moving_2D_sitk = stack_sitk_tmp[:,:,j+step]

                ## Register in 2D space
                initial_transform = sitk.CenteredTransformInitializer(fixed_2D_sitk, moving_2D_sitk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.GEOMETRY)

                registration_method = sitk.ImageRegistrationMethod()

                """
                similarity metric settings
                """
                registration_method.SetMetricAsCorrelation()

                registration_method.SetInterpolator(sitk.sitkLinear)
                
                """
                optimizer settings
                """
                registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

                registration_method.SetOptimizerScalesFromPhysicalShift()
                
                """
                setup for the multi-resolution framework            
                """
                registration_method.SetInitialTransform(initial_transform)

                rigid_transform_2D = registration_method.Execute(sitk.Cast(fixed_2D_sitk, sitk.sitkFloat64), sitk.Cast(moving_2D_sitk, sitk.sitkFloat64)) 


                ## Extract parameters of registration
                angle, translation_x, translation_y = rigid_transform_2D.GetParameters()
                center = rigid_transform_2D.GetFixedParameters()

                ## Create transformation used to align moving -> fixed
                ## Obtain inverse translation
                tmp_trafo = sitk.Euler2DTransform((0,0),-angle,(0,0))
                translation_inv = tmp_trafo.TransformPoint((-translation_x, - translation_y))

                # rigid_transform_2D_inv = sitk.AffineTransform(test.GetInverse())
                rigid_transform_2D_inv = sitk.Euler2DTransform(center, -angle, translation_inv)


                ## Get transformation for 3D in-plane rigid transformation to update T_PI of slice
                T_PP = self._get_3D_transform_to_align_stack_with_physical_coordinate_system(slice_3D.sitk)

                ## Get transformation for 3D in-plane rigid transformation
                T_PI_in_plane_rotation_3D = self._get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update slice with the obtained transformation
                slice_3D.update_affine_transform(T_PI_in_plane_rotation_3D)

                ## * Unit Test add on
                UT_warped_2D = sitk.Resample(moving_2D_sitk, fixed_2D_sitk, rigid_transform_2D, sitk.sitkLinear, 0.0, moving_2D_sitk.GetPixelIDValue())

                UT_image_nda[j+step,:,:] = sitk.GetArrayFromImage(UT_warped_2D)
                ## * ## 

            ## * Unit Test add on
            UT_image_planar = sitk.GetImageFromArray(UT_image_nda)
            UT_image_planar.CopyInformation(self._stacks[i].sitk)

            self._UT_2D_resampled_stacks[i] = UT_image_planar

            # Write file
            # full_file_name = os.path.join("results/", self._stacks[i].get_filename() + "_aligned_2D_base.nii.gz")
            # sitk.WriteImage(self._UT_2D_resampled_stacks[i], full_file_name)
            ## * ## 

        return None


    # def _resample_stacks_of_planarly_aligned_slices(self):

    #     for i in range(0, self._N_stacks):
    #         stack = self._stacks[i]
    #         slices = stack.get_slices()
    #         N_slices = stack.get_number_of_slices()

    #         warped_stack_nda = sitk.GetArrayFromImage(stack.sitk)
    #         warped_stack_mask_nda = sitk.GetArrayFromImage(stack.sitk_mask)


    #         # Identity transform since trafo is already updated in image header
    #         transform = sitk.Euler3DTransform()

    #         for j in range(0, N_slices):
    #             ## Image    
    #             fixed_sitk = stack.sitk[:,:,j:j+1]
    #             moving_sitk = slices[j].sitk

    #             warped_slice_sitk = sitk.Resample(moving_sitk, fixed_sitk, transform, sitk.sitkLinear, 0.0, moving_sitk.GetPixelIDValue())
    #             warped_slice_nda = sitk.GetArrayFromImage(warped_slice_sitk)
    #             warped_stack_nda[j,:,:] = warped_slice_nda[0,:,:]

    #             ## Mask
    #             fixed_sitk = stack.sitk_mask[:,:,j:j+1]
    #             moving_sitk = slices[j].sitk_mask

    #             warped_slice_sitk = sitk.Resample(moving_sitk, fixed_sitk, transform, sitk.sitkLinear, 0.0, moving_sitk.GetPixelIDValue())
    #             warped_slice_nda = sitk.GetArrayFromImage(warped_slice_sitk)
    #             warped_stack_mask_nda[j,:,:] = warped_slice_nda[0,:,:]


    #         ## Image
    #         warped_stack = sitk.GetImageFromArray(warped_stack_nda)
    #         warped_stack.CopyInformation(stack.sitk)

    #         ## Mask
    #         warped_stack_mask = sitk.GetImageFromArray(warped_stack_mask_nda)
    #         warped_stack_mask.CopyInformation(stack.sitk_mask)

    #         ## Update member variables
    #         # self._resampled_sitk_stacks_after_in_plane_alignment[i] = warped_stack
    #         # self._resampled_sitk_stack_masks_after_in_plane_alignment[i] = warped_stack_mask

    #         self._stacks_of_planarly_aligned_slices[i] = st.Stack.from_sitk_image(warped_stack, stack.get_filename() + "planarly_aligned_slices", warped_stack_mask)

    #     return None


    # def _UT_get_resampled_stacks(self):
    # def get_resampled_planarly_aligned_stacks(self):
    #     ## In case stacks have not been resample yet, do it
    #     if all(i is None for i in self._stacks_of_planarly_aligned_slices):
    #         self._resample_stacks_of_planarly_aligned_slices()

    #     return self._stacks_of_planarly_aligned_slices


    def write_resampled_stacks(self, directory):
        ## In case stacks have not been resample yet, do it
        if all(i is None for i in self._resampled_sitk_stacks_after_in_plane_alignment):
            self._resample_stacks_of_planarly_aligned_slices()

        ## Write the resampled stacks
        for i in range(0, self._N_stacks):
            filename = self._stacks[i].get_filename()

            ## Image
            full_file_name = os.path.join(directory, filename + "_aligned.nii.gz")
            sitk.WriteImage(self._resampled_sitk_stacks_after_in_plane_alignment[i], full_file_name)

            ## Mask
            full_file_name = os.path.join(directory, filename + "_aligned_mask.nii.gz")
            sitk.WriteImage(self._resampled_sitk_stack_masks_after_in_plane_alignment[i], full_file_name)

        return None


    """
    Functions used for SimpleITK illustrations
    """
    #callback invoked when the StartEvent happens, sets up our new data
    def start_plot():
        global metric_values, multires_iterations
        
        metric_values = []
        multires_iterations = []


    #callback invoked when the EndEvent happens, do cleanup of data and figure
    def end_plot():
        global metric_values, multires_iterations
        
        del metric_values
        del multires_iterations
        #close figure, we don't want to get a duplicate of the plot latter on
        plt.close()


    #callback invoked when the IterationEvent happens, update our data and display new figure    
    def plot_values(registration_method):
        global metric_values, multires_iterations
        
        metric_values.append(registration_method.GetMetricValue())                                       
        #clear the output area (wait=True, to reduce flickering), and plot current data
        clear_output(wait=True)
        #plot the similarity metric values
        plt.plot(metric_values, 'r')
        plt.plot(multires_iterations, [metric_values[index] for index in multires_iterations], 'b*')
        plt.xlabel('Iteration Number',fontsize=12)
        plt.ylabel('Metric Value',fontsize=12)
        plt.show()
        

    #callback invoked when the sitkMultiResolutionIterationEvent happens, update the index into the 
    #metric_values list. 
    def update_multires_iterations():
        global metric_values, multires_iterations
        multires_iterations.append(len(metric_values))
## \file InPlaneRigidRegistration.py
#  \brief  
# 
#  \author Michael Ebner
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk
import numpy as np

import matplotlib.pyplot as plt


## Import modules from src-folder
import SimpleITKHelper as sitkh


class InPlaneRigidRegistration:

    def __init__(self, stack_manager):

        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        self._resampled_sitk_stacks_after_in_plane_alignment = [None]*len(self._stacks)
        self._UT_2D_resampled_stacks = [None]*len(self._stacks)

        self._run_in_plane_rigid_registration_2D_sitk_2D()

        return None


    def _run_in_plane_rigid_registration_2D_sitk_2D(self):

        ## When running unit test
        # self._UT_apply_in_plane_rigid_registration_2D_approach_01()


        ## Apply in-plane registration on each stack separately
        for i in range(0, self._N_stacks):

            # self._apply_in_plane_rigid_registration_2D_approach_01(self._stacks[i])
            self._apply_in_plane_rigid_registration_2D_approach_02(self._stacks[i])

        return None


    ## Simpy register one slice to the previous one
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
                T_PP = slice_3D.get_transform_to_align_with_physical_coordinate_system()

                ## Get registration trafo for slices in physical 3D space
                T_PI_in_plane_rotation_3D = sitkh.get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update T_PI of slice s.t. it is aligned with slice_3D_ref
                slice_3D.set_affine_transform(T_PI_in_plane_rotation_3D)

        return None


    def _apply_in_plane_rigid_registration_2D_approach_02(self, stack):
        slices = stack.get_slices()
        N_slices = stack.get_number_of_slices()

        iterations = 2
        for iteration in range(0,iterations):
            # for j in np.concatenate((range(0, N_slices, 2),range(1, N_slices, 2))):
            for j in np.concatenate((range(1, N_slices, 2),range(0, N_slices, 2))):
                slice_3D = slices[j]

                print("Iteration %r/%r: Slice = %r/%r" %(iteration+1,iterations,j+1,N_slices))

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
                    fixed_2D_sitk_mask = slice_2D_ref_sitk_mask, 
                    moving_2D_sitk_mask = slice_3D_copy_sitk_mask[:,:,0])

                ## Get transformation for 3D in-plane rigid transformation to update T_PI of slice
                T_PP = slice_3D.get_transform_to_align_with_physical_coordinate_system()

                ## Get registration trafo for slices in physical 3D space
                T_PI_in_plane_rotation_3D = sitkh.get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update T_PI of slice s.t. it is aligned with slice_3D_ref
                slice_3D.set_affine_transform(T_PI_in_plane_rotation_3D)


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
            # average_3D_slice_sitk_mask = sitk.Image(slice_3D_next.sitk_mask)
            # average_3D_slice_sitk_mask.SetOrigin(average_3D_slice_sitk.GetOrigin())
            # average_3D_slice_sitk_mask.SetDirection(average_3D_slice_sitk.GetDirection())

            # average_2D_slice_sitk = average_3D_slice_sitk[:,:,0]
            # average_2D_slice_sitk_mask = average_3D_slice_sitk_mask[:,:,0]


            slice_3D_prev_align_mask = sitk.Image(slice_3D_prev.sitk_mask)
            slice_3D_next_align_mask = sitk.Image(slice_3D_next.sitk_mask)

            slice_3D_prev_align_mask.SetOrigin(slice_3D_prev_align.GetOrigin())
            slice_3D_prev_align_mask.SetDirection(slice_3D_prev_align.GetDirection())

            slice_3D_next_align_mask.SetOrigin(slice_3D_next_align.GetOrigin())
            slice_3D_next_align_mask.SetDirection(slice_3D_next_align.GetDirection())

            slice_2D_prev_align_mask = slice_3D_prev_align_mask[:,:,0]
            slice_2D_next_align_mask = slice_3D_next_align_mask[:,:,0]

            slice_2D_next_align_mask_warped = sitk.Resample(slice_2D_next_align_mask, slice_2D_prev_align_mask, sitk.Euler2DTransform(), sitk.sitkNearestNeighbor, 0.0, slice_2D_next_align_mask.GetPixelIDValue())


            ## Compute average
            average_2D_slice_sitk =  (slice_2D_prev_align + slice_2D_next_align_warped)/2.

            average_2D_slice_sitk_mask_nda = np.round((sitk.GetArrayFromImage(slice_2D_prev_align_mask)+sitk.GetArrayFromImage(slice_2D_next_align_mask_warped)/2.))

            average_2D_slice_sitk_mask = sitk.GetImageFromArray(average_2D_slice_sitk_mask_nda)
            average_2D_slice_sitk_mask.CopyInformation(slice_2D_prev_align_mask)

            # average_2D_slice_sitk_mask = (slice_2D_prev_align_mask + slice_2D_next_align_mask_warped)/2

            # tmp = sitk.GetArrayFromImage(average_2D_slice_sitk)
            # tmp_mask = sitk.GetArrayFromImage(average_2D_slice_sitk_mask)

            # plt.subplot(1,2,1)
            # plt.imshow(sitk.GetArrayFromImage(slice_2D_prev_align_mask), cmap="Greys_r")

            # plt.subplot(1,2,2)
            # plt.imshow(sitk.GetArrayFromImage(slice_2D_next_align_mask), cmap="Greys_r")
            # plt.show()

        return average_2D_slice_sitk, average_2D_slice_sitk_mask

        # fig = plt.figure(1)
        # # plt.suptitle("Slice %r/%r: error (norm) = %r" %(j+1,N_slices,np.linalg.norm(stacks_aligned_3D_nda[j,:,:]-stacks_aligned_2D_nda[j,:,:])))
        # plt.subplot(1,3,1)
        # plt.imshow(sitk.GetArrayFromImage(slice_2D_prev_align), cmap="Greys_r")
        # plt.axis('off')

        # plt.subplot(1,3,2)
        # plt.imshow(sitk.GetArrayFromImage(slice_2D_next_align), cmap="Greys_r")
        # plt.axis('off')
        
        # plt.subplot(1,3,3)
        # plt.imshow(sitk.GetArrayFromImage(average_slice_2D), cmap="Greys_r")
        # plt.axis('off')
        # plt.show()


    def _in_plane_rigid_registration_2D(self, fixed_2D_sitk, moving_2D_sitk, fixed_2D_sitk_mask, moving_2D_sitk_mask):

        ## Set transform
        # initial_transform = sitk.CenteredTransformInitializer(fixed_2D_sitk, moving_2D_sitk, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)
        initial_transform = sitk.Euler2DTransform()

        registration_method = sitk.ImageRegistrationMethod()

        """
        similarity metric settings
        """
        # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5) #set unsigned int radius
        registration_method.SetMetricAsCorrelation()
        # registration_method.SetMetricAsDemons()
        # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=50, varianceForJointPDFSmoothing=3)
        # registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
        # registration_method.SetMetricAsMeanSquares()

        registration_method.SetMetricFixedMask(fixed_2D_sitk_mask)
        registration_method.SetMetricMovingMask(moving_2D_sitk_mask)

        # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

        registration_method.SetInterpolator(sitk.sitkLinear)
        
        """
        optimizer settings
        """
        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)
        registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        # registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        """
        setup for the multi-resolution framework            
        """
        # registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        # registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        # registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform)

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

        ## Extract parameters of 2D registration
        angle, translation_x, translation_y = rigid_transform_2D.GetParameters()
        center = rigid_transform_2D.GetFixedParameters()

        ## Create transformation used to align moving -> fixed

        ## Obtain inverse translation
        tmp_trafo = sitk.Euler2DTransform((0,0),-angle,(0,0))
        translation_inv = tmp_trafo.TransformPoint((-translation_x, - translation_y))


        # rigid_transform_2D_inv = sitk.AffineTransform(test.GetInverse())
        ## math.stackexchange.com/questions/1234948/inverse-of-a-rigid-transformation
        rigid_transform_2D_inv = sitk.Euler2DTransform(center, -angle, translation_inv)

        return rigid_transform_2D_inv


    def _get_copy_of_sitk_slice_with_aligned_physical_coordinate_system(self, slice_3D):

        ## Create copies of fixed and moving slices
        slice_3D_sitk_copy = sitk.Image(slice_3D.sitk)

        ## Get current transform from image to physical space of slice
        T_PI = sitkh.get_sitk_affine_transform_from_sitk_image(slice_3D.sitk)

        ## Get transform to get alignment with physical coordinate system of original/untouched slice
        T_PP = slice_3D.get_transform_to_align_with_physical_coordinate_system()

        ## Get transform to align slice with physical coordinate system (perhaps already shifted there) 
        T_PI_align = sitkh.get_composited_sitk_affine_transform(T_PP, T_PI)

        ## Set direction and origin of image accordingly
        direction = sitkh.get_sitk_image_direction_matrix_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)
        origin = sitkh.get_sitk_image_origin_from_sitk_affine_transform(T_PI_align, slice_3D.sitk)

        slice_3D_sitk_copy.SetDirection(direction)
        slice_3D_sitk_copy.SetOrigin(origin)

        return slice_3D_sitk_copy


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
                T_PP = slice_3D.get_transform_to_align_with_physical_coordinate_system()

                ## Get transformation for 3D in-plane rigid transformation
                T_PI_in_plane_rotation_3D = sitkh.get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D_inv, T_PP, slice_3D.sitk)

                ## Update slice with the obtained transformation
                slice_3D.set_affine_transform(T_PI_in_plane_rotation_3D)

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


    def _resample_stacks_of_aligned_slices(self):

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            warped_stack_nda = sitk.GetArrayFromImage(stack.sitk)

            for j in range(0, N_slices):
                fixed_sitk = stack.sitk[:,:,j:j+1]
                moving_sitk = slices[j].sitk

                # Identity transform since trafo is already updated in image header
                transform = sitk.Euler3DTransform()

                warped_slice_sitk = sitk.Resample(moving_sitk, fixed_sitk, transform, sitk.sitkLinear, 0.0, moving_sitk.GetPixelIDValue())

                warped_slice_nda = sitk.GetArrayFromImage(warped_slice_sitk)

                warped_stack_nda[j,:,:] = warped_slice_nda[0,:,:]


            warped_stack = sitk.GetImageFromArray(warped_stack_nda)
            warped_stack.CopyInformation(stack.sitk)

            self._resampled_sitk_stacks_after_in_plane_alignment[i] = warped_stack

        return None


    def _UT_get_resampled_stacks(self):
        ## In case stacks have not been resample yet, do it
        if all(i is None for i in self._resampled_sitk_stacks_after_in_plane_alignment):
            self._resample_stacks_of_aligned_slices()

        return self._resampled_sitk_stacks_after_in_plane_alignment


    def write_resampled_stacks(self, directory):
        ## In case stacks have not been resample yet, do it
        if all(i is None for i in self._resampled_sitk_stacks_after_in_plane_alignment):
            self._resample_stacks_of_aligned_slices()

        ## Write the resampled stacks
        for i in range(0, self._N_stacks):
            filename = self._stacks[i].get_filename()

            full_file_name = os.path.join(directory, filename + "_aligned.nii.gz")
            sitk.WriteImage(self._resampled_sitk_stacks_after_in_plane_alignment[i], full_file_name)

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
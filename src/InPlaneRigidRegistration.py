## \file InPlaneRigidRegistration.py
#  \brief  
# 
#  \author Michael Ebner
#  \date September 2015


## Import libraries
import os                       # used to execute terminal commands in python
import SimpleITK as sitk

## Import modules from src-folder
import SimpleITKHelper as sitkh


class InPlaneRigidRegistration:

    def __init__(self, stack_manager):

        self._stacks = stack_manager.get_stacks()
        self._N_stacks = stack_manager.get_number_of_stacks()

        self._run_in_plane_rigid_registration()

        return None


    def _run_in_plane_rigid_registration(self):

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            step = 1

            test = sitk.ReadImage(stack.get_directory()+stack.get_filename()+".nii.gz", sitk.sitkFloat32)
            test_planar = sitk.ReadImage(stack.get_directory()+stack.get_filename()+".nii.gz", sitk.sitkFloat32)

            test_nda = sitk.GetArrayFromImage(test)

            for j in range(0, N_slices-step):
            # for j in range(0, 10):
                fixed_3D = slices[j+step]
                moving_3D = slices[j+step]

                fixed_2D_sitk = slices[j].sitk[:,:,0]
                moving_2D_sitk = slices[j+step].sitk[:,:,0]




                rigid_transform_2D = self._in_plane_rigid_registration(fixed_2D_sitk, moving_2D_sitk)

                warped_2D = sitk.Resample(moving_2D_sitk, fixed_2D_sitk, rigid_transform_2D, sitk.sitkLinear, 0.0, moving_2D_sitk.GetPixelIDValue())


                test_nda[j+step,:,:] = sitk.GetArrayFromImage(warped_2D)
                test_planar = sitk.GetImageFromArray(test_nda)
                test_planar.CopyInformation(self._stacks[i].sitk)




                rigid_transform_3D = sitkh.get_3D_in_plane_alignment_transform_from_sitk_2D_rigid_transform(rigid_transform_2D, fixed_3D.sitk)


                transform = self._update_affine_transform(fixed_3D, moving_3D, rigid_transform_3D)
                fixed_3D.set_affine_transform(transform)

                print("Iteration " + str(j+1) + "/" + str(N_slices-1) + ":")

            full_file_name = os.path.join("../results/", self._stacks[i].get_filename() + "_aligned_2D_base.nii.gz")
            sitk.WriteImage(test_planar, full_file_name)

        return None


    def _in_plane_rigid_registration(self, fixed, moving):

        initial_transform = sitk.CenteredTransformInitializer(fixed, moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

        registration_method = sitk.ImageRegistrationMethod()

        """
        similarity metric settings
        """
        # registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5) #set unsigned int radius
        # registration_method.SetMetricAsCorrelation()
        # registration_method.SetMetricAsDemons()
        # registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20, varianceForJointPDFSmoothing=1.5)
        registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=100)
        # registration_method.SetMetricAsMeanSquares()

        # registration_method.SetMetricFixedMask(fixed_mask)
        # registration_method.SetMetricMovingMask(moving_mask)
        # registration_method.SetMetricSamplingStrategy(registration_method.NONE)

        registration_method.SetInterpolator(sitk.sitkLinear)
        
        """
        optimizer settings
        """
        # registration_method.SetOptimizerAsConjugateGradientLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-8, convergenceWindowSize=10)
        # registration_method.SetOptimizerAsGradientDescentLineSearch(learningRate=1, numberOfIterations=100, convergenceMinimumValue=1e-6, convergenceWindowSize=10)
        registration_method.SetOptimizerAsRegularStepGradientDescent(learningRate=1, minStep=1, numberOfIterations=100)

        registration_method.SetOptimizerScalesFromPhysicalShift()
        
        """
        setup for the multi-resolution framework            
        """
        registration_method.SetShrinkFactorsPerLevel(shrinkFactors = [4,2,1])
        registration_method.SetSmoothingSigmasPerLevel(smoothingSigmas=[2,1,0])
        registration_method.SmoothingSigmasAreSpecifiedInPhysicalUnitsOn()

        registration_method.SetInitialTransform(initial_transform)

        #connect all of the observers so that we can perform plotting during registration
        # registration_method.AddCommand(sitk.sitkStartEvent, start_plot)
        # registration_method.AddCommand(sitk.sitkEndEvent, end_plot)
        # registration_method.AddCommand(sitk.sitkMultiResolutionIterationEvent, update_multires_iterations) 
        # registration_method.AddCommand(sitk.sitkIterationEvent, lambda: plot_values(registration_method))

        # print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
        # print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
        # print("\n")

        final_transform_2D = registration_method.Execute(sitk.Cast(fixed, sitk.sitkFloat32), sitk.Cast(moving, sitk.sitkFloat32)) 

        return final_transform_2D


    def _update_affine_transform(self, fixed_3D, moving_3D, rigid_transform_3D):
        transform_moving = moving_3D.get_affine_transform()
        transform_fixed = fixed_3D.get_affine_transform()

        transform = sitkh.get_composited_sitk_affine_transform(rigid_transform_3D, transform_fixed)
        # transform = sitkh.get_composited_sitk_affine_transform(transform, transform_moving)

        return transform


    def write_resampled_stacks(self, directory):

        for i in range(0, self._N_stacks):
            stack = self._stacks[i]
            slices = stack.get_slices()
            N_slices = stack.get_number_of_slices()

            warped_stack_nda = sitk.GetArrayFromImage(stack.sitk)

            for j in range(0, N_slices):
                slice_sitk = slices[j].sitk
                transform = slices[j].get_affine_transform()
                warped_slice = sitk.Resample(slice_sitk, transform, sitk.sitkLinear, 0.0, slice_sitk.GetPixelIDValue())

                slice_nda = sitk.GetArrayFromImage(warped_slice)

                warped_stack_nda[j,:,:] = slice_nda[0,:,:]

            warped_stack = sitk.GetImageFromArray(warped_stack_nda)
            warped_stack.CopyInformation(stack.sitk)

            full_file_name = os.path.join(directory, stack.get_filename() + "_aligned.nii.gz")
            sitk.WriteImage(warped_stack, full_file_name)

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
import SimpleITK as sitk
import numpy as np

import sys
sys.path.append("../v1_20150915/")

from FileAndImageHelpers import *

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


"""
Use SimpleITK to register images in-plane
"""

dir_input = "../../results/input_data/"
dir_output = "results/"
filename =  "0"


stack = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)
# stack_aligned_planar = sitk.ReadImage(dir_input+filename+".nii.gz", sitk.sitkFloat32)

stack_mask = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)
# stack_mask_aligned_planar = sitk.ReadImage(dir_input+filename+"_mask.nii.gz", sitk.sitkUInt8)

N = stack.GetSize()[-1]


i = 0
step = 1

slice_3D_fixed = stack[:,:,i:i+1]
slice_3D_moving = stack[:,:,i+step:i+step+1]

slice_2D_fixed = slice_3D_fixed[:,:,0]
slice_2D_moving = slice_3D_moving[:,:,0]


# stack_nda = sitk.GetArrayFromImage(stack_aligned_planar) #now indexed as [z,y,x]!
# stack_mask_nda = sitk.GetArrayFromImage(stack_mask_aligned_planar) #now indexed as [z,y,x]!

"""
Register slices in-plane:
"""

initial_transform = sitk.CenteredTransformInitializer(
    slice_2D_fixed, slice_2D_moving, sitk.Euler2DTransform(), sitk.CenteredTransformInitializerFilter.MOMENTS)

registration_method = sitk.ImageRegistrationMethod()

"""
similarity metric settings
"""
registration_method.SetMetricAsANTSNeighborhoodCorrelation(radius=5) #set unsigned int radius
# registration_method.SetMetricAsCorrelation()
# registration_method.SetMetricAsDemons()
# registration_method.SetMetricAsJointHistogramMutualInformation(numberOfHistogramBins=20, varianceForJointPDFSmoothing=1.5)
# registration_method.SetMetricAsMattesMutualInformation(numberOfHistogramBins=50)
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

final_transform = registration_method.Execute(
    sitk.Cast(slice_2D_fixed, sitk.sitkFloat32), sitk.Cast(slice_2D_moving, sitk.sitkFloat32))

angle_z, translation_x, translation_y = final_transform.GetParameters()
center = final_transform.GetFixedParameters()


rigid_transform_3D = sitk.Euler3DTransform()
rigid_transform_3D.SetRotation(0, 0, angle_z)
rigid_transform_3D.SetTranslation((translation_x, translation_y, 0))
rigid_transform_3D.SetCenter((center[0], center[1], 0))

A = rigid_transform_3D.GetMatrix()
t = rigid_transform_3D.GetTranslation()



# composite_transform = sitk.Transform(rigid_transform_3D) # not necessary here but now trafos can be composited!!


warped = sitk.Resample(slice_2D_moving, slice_2D_fixed, final_transform, sitk.sitkLinear, 0.0, slice_2D_moving.GetPixelIDValue())
# warped_mask = sitk.Resample(moving_mask, fixed_mask, final_transform, sitk.sitkNearestNeighbor, 0.0, moving_mask.GetPixelIDValue())

# # for i in xrange(0,N-1):
# for i in xrange(35,36):
#     if i == N-2:
#         step=1

#     fixed = stack_aligned_planar[:,:,i]
#     moving = stack_aligned_planar[:,:,i+step]

#     fixed_mask = stack_mask_aligned_planar[:,:,i]
#     moving_mask = stack_mask_aligned_planar[:,:,i+step]

    

   

#     stack_nda[i+step,:,:] = sitk.GetArrayFromImage(warped)
#     stack_mask_nda[i+step,:,:] = sitk.GetArrayFromImage(warped_mask)

#     # plot_comparison_of_reference_and_warped_image(
#         # fixed=sitk.GetArrayFromImage(fixed), warped=sitk.GetArrayFromImage(warped), fig_id=2)

#     # myshow(fixed-moving)

#     print("Iteration " + str(i+1) + "/" + str(N-1) + ":")
#     print('  Final metric value: {0}'.format(registration_method.GetMetricValue()))
#     print('  Optimizer\'s stopping condition, {0}'.format(registration_method.GetOptimizerStopConditionDescription()))
#     print("\n")

#     fig = plot_comparison_of_reference_and_warped_image(
#         fixed=sitk.GetArrayFromImage(stack)[i,:,:]-sitk.GetArrayFromImage(stack)[i+1,:,:], 
#         warped=sitk.GetArrayFromImage(fixed)-sitk.GetArrayFromImage(warped),
#         fixed_title="without registration between Slice " + str(i) + " and " + str(i+step),
#         warped_title="rigidly registered",
#         fig_id=1)
#     fig.canvas.draw()

#     fig = plot_comparison_of_reference_and_warped_image(
#         fixed=sitk.GetArrayFromImage(stack_mask)[i,:,:]-sitk.GetArrayFromImage(stack_mask)[i+1,:,:], 
#         warped=sitk.GetArrayFromImage(fixed_mask)-sitk.GetArrayFromImage(warped_mask),
#         fixed_title="without registration between Slice " + str(i) + " and " + str(i+step),
#         warped_title="rigidly registered",
#         fig_id=2)
#     fig.canvas.draw()
#     # time.sleep(1)

#     stack_aligned_planar = sitk.GetImageFromArray(stack_nda)
#     stack_aligned_planar.CopyInformation(stack)

#     stack_mask_aligned_planar = sitk.GetImageFromArray(stack_mask_nda)
#     stack_mask_aligned_planar.CopyInformation(stack_mask)

# # imshow(sitk.GetArrayFromImage(fixed)-sitk.GetArrayFromImage(moving),cmap=cm.gray)

# # myshow(fixed-warped)

# sitk.WriteImage(stack_aligned_planar, os.path.join(dir_output,filename+"_aligned_planar.nii.gz"))
# sitk.WriteImage(stack_mask_aligned_planar, os.path.join(dir_output,filename+"_mask_aligned_planar.nii.gz"))

# plt.show()